/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/transforms/memory_plan.cc
 * \brief A pass for manifesting explicit memory allocations.
 */

#include <dmlc/common.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../backend/compile_engine.h"
#include "let_list.h"

using namespace tvm::runtime;

namespace tvm {
namespace relay {

using RegionMap = std::unordered_map<Var, std::pair<Expr, Expr>, ObjectPtrHash, ObjectPtrEqual>;

inline Constant MakeConstant(int64_t value) {
  auto tensor = NDArray::Empty({}, {kDLInt, 64, 1}, {kDLCPU, 0});
  reinterpret_cast<int64_t*>(tensor->data)[0] = value;
  return std::move(Constant(tensor));
}

inline TVMContext EmptyContext() {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(-1);
  ctx.device_id = -1;
  return ctx;
}

inline bool ValidContext(const TVMContext& ctx) {
  return static_cast<int>(ctx.device_type) > 0 && ctx.device_id >= 0;
}

Expr IterativeLet(
    const Expr& let, const std::function<std::pair<Expr, Expr>(const Expr&, const Expr&)>& each_binding,
    const std::function<Expr(const std::vector<std::pair<Expr, Expr>>&, const Expr&)>& kont) {
  std::vector<std::pair<Expr, Expr>> bindings;
  const auto* ln = let.as<LetNode>();
  while (ln) {
    Var lhs = ln->var;
    Expr rhs = ln->value;
    bindings.push_back(each_binding(lhs, rhs));
    ln = ln->body.as<LetNode>();
  }

  return kont(bindings, let);
}

Expr MakeLet(const std::vector<std::pair<Expr, Expr>>& bindings, Expr body) {
  for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
    CHECK(it->first.defined());
    CHECK(it->second.defined());
    CHECK(body.defined());
    body = Let(Downcast<Var>(it->first), it->second, body);
  }
  return body;
}

class Region {
 public:
  Region(const Var& var, const Expr& size, const Expr& alignment, const DataType& dtype,
         const TVMContext& ctx, const RegionMap& offsets)
      : var_(var),
        size_(size),
        alignment_(alignment),
        dtype_(dtype),
        ctx_(ctx),
        offsets_(offsets) {}

  static std::shared_ptr<Region> Empty(int region_no) {
    Expr zero = MakeConstant(0);
    Var region_var("region" + std::to_string(region_no), Type(nullptr));
    return std::make_shared<Region>(
        Region(region_var, zero, Expr(nullptr), DataType::Void(), EmptyContext(), {}));
  }

  // Grow the region by a given allocation as well as track the old storage
  // for later rewriting the program to use the allocated region.
  void Grow(const Var& old_storage, const Expr& size, const Expr& alignment, const TVMContext& ctx,
            const DataType& dtype) {
    if (!dtype_.is_void()) {
      CHECK(dtype_ == dtype) << "must have matching dtypes in a region";
    } else {
      dtype_ = dtype;
    }

    if (alignment_.defined()) {
      StructuralEqual eq;
      CHECK(eq(alignment_, alignment)) << "must have matching alignments in a region";
    } else {
      alignment_ = alignment;
    }

    if (ValidContext(ctx_)) {
      CHECK(ctx_.device_type == ctx.device_type && ctx_.device_id == ctx.device_id)
          << "must have mathcing contexg";
    } else {
      CHECK(ValidContext(ctx));
      ctx_ = ctx;
    }

    auto fa = runtime::Registry::Get("relay.op._make.add");
    CHECK(fa) << "cannot find operator add from the registry";
    auto fs = runtime::Registry::Get("relay.op._make.subtract");
    CHECK(fs) << "cannot find operator subtract from the registry";
    auto fm = runtime::Registry::Get("relay.op._make.multiply");
    CHECK(fa) << "cannot find operator multiply from the registry";
    auto fd = runtime::Registry::Get("relay.op._make.divide");
    CHECK(fd) << "cannot find operator devide from the registry";
    Expr add = (*fa)(size, alignment_);
    Expr sub = (*fs)(add, MakeConstant(1));
    Expr div = (*fd)(sub, alignment_);
    Expr new_size = (*fm)(div, alignment_);

    Var offset_var("offset" + std::to_string(offsets_.size()), Type(nullptr));
    offsets_[old_storage] = std::make_pair(offset_var, size_);
    size_ = (*fa)(size_, new_size);
  }

  Expr OffsetFor(const Var& alloc) const {
    auto it = offsets_.find(alloc);
    if (it == offsets_.end()) {
      return Expr(nullptr);
    } else {
      return it->second.first;
    }
  }

  // Generate the prelude code for a region, wrapping the body in it.
  //
  // The prelude contains the single allocation for a region, and
  // all offset computations.
  Expr ToExpr(Expr body) {
    if (!ValidContext(ctx_)) {
      ctx_.device_type = kDLCPU;
      ctx_.device_id = 0;
    }

    // Generate bindings for each and every size computation
    // we must do this to maintain ANF.
    std::vector<std::pair<Expr, Expr>> bindings;

    // First compute the total size
    StructuralHash hash;
    Var total_size("total_size" + std::to_string(hash(body)), Type(nullptr));
    bindings.push_back(std::make_pair(total_size, size_));

    // Allocate the entire region with a single call.
    auto alloc_storage = runtime::Registry::Get("relay.op.memory._make.alloc_storage");
    Expr alloc = (*alloc_storage)(total_size, alignment_, ctx_, dtype_);
    bindings.push_back(std::make_pair(var_, alloc));

    // Generate variables which contain all of the offset math.
    // Ensure we constant evaluate away all the math here.
    //
    // In theory we can support dynamic offsets but this
    // requires another round of memory planning and
    // potentially coalescing.
    for (const auto& alloc : offsets_) {
      bindings.push_back(alloc.second);
    }

    body = MakeLet(bindings, body) ;
    return body;
  }

 private:
  Var var_;
  Expr size_;
  Expr alignment_;
  DataType dtype_;
  TVMContext ctx_;
  RegionMap offsets_;

  friend class StorageCoalesce;
};

using RegionPtr = std::shared_ptr<Region>;

// A pass for coalescing allocations into region/arena allocations.
//
// After this pass each allocation comes from the same backing storage,
// but will never overlap even in time, i.e. the allocations are just
// packed into a contiguous block of memory.
//
// A secondary part of memory planning will perform liveness analysis to
// overlap these in time, i.e when an early tensor dies we will attempt
// to reuse its slot.
class StorageCoalesce : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionNode* fn) final {
    if (fn->HasNonzeroAttr(attr::kPrimitive)) {
      return ExprMutator::VisitExpr_(fn);
    } else {
      EnterScope();
      Expr body = ExprMutator::Mutate(fn->body);
      body = ExitScope(body);
      return Function(fn->params, body, fn->ret_type, fn->type_params, fn->attrs);
    }
  }

  Expr VisitExpr_(const IfNode* in) final {
    EnterScope();
    Expr true_branch = ExprMutator::Mutate(in->true_branch);
    true_branch = ExitScope(true_branch);

    EnterScope();
    Expr false_branch = ExprMutator::Mutate(in->false_branch);
    false_branch = ExitScope(false_branch);

    return If(in->cond, true_branch, false_branch);
  }

  Expr VisitExpr_(const LetNode* ln) final {
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> dynamic_regions;

    auto each_binding = [&](const Expr& lhs, const Expr& rhs) {
      if (const CallNode* rhs_call = rhs.as<CallNode>()) {
        if (rhs_call->op == Op::Get("memory.alloc_storage")) {
          return ProcessAllocStorage(lhs, rhs_call, &dynamic_regions);
        } else if (rhs_call->op == Op::Get("memory.alloc_tensor")) {
          return ProcessAllocTensor(lhs, rhs_call);
        }
      }
      return std::make_pair(lhs, rhs);
    };

    Expr res = IterativeLet(GetRef<Let>(ln), each_binding, MakeLet(dynamic_regions));
    CHECK(res.defined());
    return res;
  }

  std::pair<Expr, Expr> ProcessAllocStorage(
      const Expr& lhs, const CallNode* call,
      std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>* dynamic_regions) {
    Expr size = call->args[0];
    Expr alignment = call->args[1];

    const auto* alloc_attrs = call->attrs.as<AllocStorageAttrs>();
    CHECK(alloc_attrs);
    DataType dtype = alloc_attrs->dtype;
    TVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(alloc_attrs->device_type);
    ctx.device_id = alloc_attrs->device_id;

    if (!size->IsInstance<ConstantNode>()) {
      EnterScope();
      dynamic_regions->emplace(lhs);
    } else {
      // A new scope is created when entering a new region with different
      // device context
      RegionPtr region = CurrentRegion(dtype);
      if (ValidContext(region->ctx_) &&
          (region->ctx_.device_type != ctx.device_type || region->ctx_.device_id != ctx.device_id)) {
        EnterScope();
        dynamic_regions->emplace(lhs);
      }
    }
    RegionPtr reg = CurrentRegion(dtype);
    CHECK(lhs->IsInstance<VarNode>());
    reg->Grow(Downcast<Var>(lhs), size, alignment, ctx, dtype);
    return std::make_pair(lhs, reg->var_);
  }

  std::pair<Expr, Expr> ProcessAllocTensor(const Expr& lhs, const CallNode* call) {
    Expr storage = call->args[0];
    CHECK(storage->IsInstance<VarNode>());
    Expr old_offset = call->args[1];
    Expr shape = call->args[2];
    auto reg_offset = NewRegionAndOffset(Downcast<Var>(storage));
    RegionPtr reg = reg_offset.first;
    Expr offset = reg_offset.second;
    Call new_call(call->op, {reg->var_, offset, shape}, call->attrs);
    return std::make_pair(lhs, new_call);
  }

 private:
  struct Hash {
    size_t operator()(const DataType& dtype) const {
      size_t const h1(std::hash<int>()(dtype.code()));
      size_t const h2(std::hash<int>()(dtype.bits()));
      size_t const h3(std::hash<int>()(dtype.lanes()));
      return dmlc::HashCombine(h1, dmlc::HashCombine(h2, h3));
    }
  };

  DataType DefaultDtype() const {
    DLDataType dtype = {.code = 255, .bits = 255, .lanes = 255};
    return DataType(dtype);
  }

  // Let bind the dynamic regions
  std::function<Expr(const std::vector<std::pair<Expr, Expr>>&, const Expr&)> MakeLet(
      const std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>& dynamic_regions) {
    return [=](const std::vector<std::pair<Expr, Expr>>& bindings, Expr body) {
      for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
        Expr var = it->first;
        Expr val = it->second;
        CHECK(var.defined() && val.defined());
        body = Let(Downcast<Var>(var), val, body);
        if (dynamic_regions.find(var) != dynamic_regions.end()) {
          body = ExitScope(body);
        }
      }
      return body;
    };
  }

  void EnterScope() {
    std::unordered_map<DataType, RegionPtr, Hash> regions;
    auto default_dtype = DefaultDtype();
    regions.emplace(default_dtype, Region::Empty(regions_.size()));
    regions_.push_back(regions);
  }

  // When leaving a scope build a region allocation for the scope.
  Expr ExitScope(Expr body) {
    auto dtype_region = regions_.back();

    for (const auto& it : dtype_region) {
      auto region = it.second;
      if (!region->offsets_.empty()) {
        body = region->ToExpr(body);
      }
    }
    regions_.pop_back();
    return body;
  }

  RegionPtr GetRegion(const DataType& dtype,
                      const std::unordered_map<DataType, RegionPtr, Hash>& regions) const {
    const auto& it = regions.find(dtype);
    if (it == regions.end()) {
      DataType default_dtype = DefaultDtype();
      CHECK_GT(regions.count(default_dtype), 0);
      return regions.at(default_dtype);
    } else {
      return it->second;
    }
  }

  RegionPtr CurrentRegion(const DataType& dtype) const {
    const auto& current_scope = regions_.back();
    return GetRegion(dtype, current_scope);
  }

  std::pair<RegionPtr, Expr> NewRegionAndOffset(const Var& old_storage) const {
    for (auto it = regions_.rbegin(); it != regions_.rend(); ++it) {
      for (const auto& dtype : *it) {
        auto region = GetRegion(dtype.first, *it);
        auto offset = region->OffsetFor(old_storage);
        if (offset.defined()) {
          return std::make_pair(region, offset);
        }
      }
    }
    LOG(FATAL) << "Could not find offset in any valid region";
    return std::make_pair(Region::Empty(0), Expr(nullptr));
  }

 private:
  std::vector<std::unordered_map<DataType, RegionPtr, Hash>> regions_;
};

namespace transform {

Pass MemoryPlan1() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(StorageCoalesce().Mutate(f));
      };
  Pass memory_plan = CreateFunctionPass(pass_func, 0, "FoldConstant", {});
  return Sequential({memory_plan, InferType()}, "MemoryPlan");
}

TVM_REGISTER_GLOBAL("relay.transform.MemoryPlan1").set_body_typed(MemoryPlan1);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
