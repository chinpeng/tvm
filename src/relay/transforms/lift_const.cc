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
 * \file src/relay/transforms/lift_const.cc
 * \brief A pass that lifts the constant to the top level of a function.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

extern Expr MakeLet(const std::vector<std::pair<Expr, Expr>>& bindings, Expr body);

class ConstLifter : public ExprMutator {
 public:
  Expr VisitExpr_(const ConstantNode* cn) final {
    Var var("const" + std::to_string(const_id_), Type(nullptr));
    const_id_++;
    constants_.push_back(std::make_pair(var, GetRef<Expr>(cn)));
    return var;
  }

  Expr VisitExpr_(const FunctionNode* fn) final {
    if (fn->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Function>(fn);
    }

    auto outer_constant = constants_;
    constants_.clear();
    // Populates constants_
    Expr body = ExprMutator::Mutate(fn->body);
    body = MakeLet(constants_, body);
    constants_ = outer_constant;

    return Function(fn->params, body, fn->ret_type, fn->type_params, fn->attrs);
  }

  Expr VisitExpr_(const LetNode* ln) final {
    std::vector<std::pair<Expr, Expr>> bindings;
    const LetNode* let = ln;
    while (let) {
      Var new_var = let->var;
      Expr new_val = let->value;
      bindings.push_back(std::make_pair(new_var, new_val));
      let = let->body.as<LetNode>();
    }
    Expr new_body = ExprMutator::Mutate(GetRef<Let>(let));
    return MakeLet(bindings, new_body);
  }

 private:
  int const_id_{0};
  bool top_level_{true};
  std::vector<std::pair<Expr, Expr>> constants_;
};

namespace transform {

Pass LiftConstants() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ConstLifter().Mutate(f));
      };
  return CreateFunctionPass(pass_func, 0, "LiftConstants", {});
}

TVM_REGISTER_GLOBAL("relay.transform.LiftConstatns").set_body_typed(LiftConstants);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
