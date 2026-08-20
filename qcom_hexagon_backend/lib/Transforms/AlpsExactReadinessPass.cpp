//===- AlpsExactReadinessPass.cpp - ALPS P3a contract audit -------------===//

#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <string>

using namespace mlir;
using namespace mlir::omni_fetch;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSEXACTREADINESS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct AlpsExactReadinessPass final
    : ::impl::AlpsExactReadinessBase<AlpsExactReadinessPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    int64_t asyncCandidates = 0;
    int64_t exactContracts = 0;
    int64_t rejected = 0;
    std::string records;
    llvm::raw_string_ostream record(records);

    function.walk([&](PrefetchInSituOp prefetch) {
      if (prefetch.getLookahead() <= 0 ||
          prefetch.getLayoutTransform() == LayoutTransform::L2Hint)
        return;
      ++asyncCandidates;
      auto action = prefetch->getAttrOfType<StringAttr>("alps.p2d.action");
      StringRef reason = "not_p2d_dma_admitted";
      bool accepted = action && action.getValue() == "dma_vtcm_async";
      if (accepted && !prefetch->getParentOfType<scf::ForOp>()) {
        accepted = false;
        reason = "missing_tile_loop";
      } else if (accepted && prefetch.getTileParams().empty()) {
        accepted = false;
        reason = "missing_exact_tile_identity";
      } else if (accepted) {
        reason = "exact_descriptor_contract";
      }

      Builder builder(function.getContext());
      prefetch->setAttr("alps.p3a.exact_readiness",
                        builder.getBoolAttr(accepted));
      prefetch->setAttr("alps.p3a.reason", builder.getStringAttr(reason));
      prefetch->setAttr(
          "alps.p3a.identity",
          builder.getStringAttr(
              "invocation_generation+value_version+tile+layout+tiers+slot_generation"));
      if (accepted)
        ++exactContracts;
      else
        ++rejected;
      record << "[ALPS-P3A-SITE] function=" << function.getName()
             << " layout=" << static_cast<int32_t>(prefetch.getLayoutTransform())
             << " lookahead=" << prefetch.getLookahead()
             << " accepted=" << accepted << " reason=" << reason << '\n';
    });
    function.walk([&](ExactWeightKickOp kick) {
      ++asyncCandidates;
      ++exactContracts;
      kick->setAttr("alps.p3a.exact_readiness",
                    Builder(function.getContext()).getBoolAttr(true));
      record << "[ALPS-P3A-SITE] function=" << function.getName()
             << " layout=1 lookahead=1 accepted=1"
             << " reason=descriptor_bound_exact_weight\n";
    });

    Builder builder(function.getContext());
    function->setAttr("alps.p3a.async_candidates",
                      builder.getI64IntegerAttr(asyncCandidates));
    function->setAttr("alps.p3a.exact_contracts",
                      builder.getI64IntegerAttr(exactContracts));
    function->setAttr("alps.p3a.rejected",
                      builder.getI64IntegerAttr(rejected));
    record << "[ALPS-P3A-SUMMARY] function=" << function.getName()
           << " async_candidates=" << asyncCandidates
           << " exact_contracts=" << exactContracts
           << " rejected=" << rejected << '\n';
    record.flush();
    static std::mutex outputMutex;
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::errs() << records;
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createAlpsExactReadinessPass() {
  return std::make_unique<AlpsExactReadinessPass>();
}
