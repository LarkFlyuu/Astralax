#include "Astralax/Helper/Helper.h"

using namespace mlir;

namespace astl {

void outputCode(const ModuleOp &module, const std::string &filenameWithExt) {
  if (module == nullptr) {
    llvm::report_fatal_error(llvm::StringRef("invalid module write to file"));
  }
  
  OpPrintingFlags flags;
  flags.enableDebugInfo();
  std::string errorMessage;
  auto output = openOutputFile(filenameWithExt, &errorMessage);
  
  if (!output) {
    llvm::report_fatal_error(llvm::StringRef(errorMessage));
  }

  module->print(output->os(), flags);
  output->keep();
}

void outputBinary(const ModuleOp &module, const std::string &output) {
  std::error_code ec;
  llvm::raw_fd_ostream dest(output, ec, llvm::sys::fs::OF_None);

  if (ec) llvm::report_fatal_error("Could not open output file");

  mlir::FallbackAsmResourceMap fallbackResourceMap;
  mlir::ParserConfig config(module->getContext(), true, &fallbackResourceMap);
  mlir::BytecodeWriterConfig writerConfig(fallbackResourceMap);

  if (mlir::writeBytecodeToFile(module, dest, writerConfig).failed())
    llvm::report_fatal_error("Write binary file failed");
  dest.flush();
}

NameLoc getLoc(MLIRContext *context, const std::string &name) {
  return NameLoc::get(StringAttr::get(context, name));
}

} // namespace astl