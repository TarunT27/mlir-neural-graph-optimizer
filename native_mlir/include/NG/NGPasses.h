#ifndef NG_NGPASSES_H
#define NG_NGPASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace ng {

std::unique_ptr<Pass> createConstantFoldPass();
std::unique_ptr<Pass> createCanonicalizePass();
std::unique_ptr<Pass> createFuseLinearReluPass();
std::unique_ptr<Pass> createDCEPass();
std::unique_ptr<Pass> createResourceGuardPass();
std::unique_ptr<Pass> createConvertNGToLinalgPass();
void registerNGPasses();

} // namespace ng
} // namespace mlir

#endif // NG_NGPASSES_H
