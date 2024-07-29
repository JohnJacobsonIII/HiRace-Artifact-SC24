//==============================================================================
// FILE:
//    CudaRWMain.cpp
//
// DESCRIPTION:
//    A standalone tool that runs the CudaRW plugin. See
//    CudaRW.cpp for a complete description.
//
// USAGE:
//    * jj-cuda-rw input-file.cpp
//
// REFERENCES:
//    Based on an example by Peter Smith:
//      * https://s3.amazonaws.com/connect.linaro.org/yvr18/presentations/yvr18-223.pdf
//
// License: The Unlicense
//==============================================================================
#include "CudaRW.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Command line options
//===----------------------------------------------------------------------===//
static llvm::cl::OptionCategory CRWCategory("cudarw options");

//===----------------------------------------------------------------------===//
// PluginASTAction
//===----------------------------------------------------------------------===//
class CRWPluginAction : public PluginASTAction {
public:
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    CRWRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<CudaRWASTConsumer>(CRWRewriter);
  }

private:
  Rewriter CRWRewriter;
};

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
int main(int Argc, const char **Argv) {
  Expected<tooling::CommonOptionsParser> eOptParser =
      clang::tooling::CommonOptionsParser::create(Argc, Argv, CRWCategory);
  if (auto E = eOptParser.takeError()) {
    errs() << "Problem constructing CommonOptionsParser "
           << toString(std::move(E)) << '\n';
    return EXIT_FAILURE;
  }
  clang::tooling::ClangTool Tool(eOptParser->getCompilations(),
                                 eOptParser->getSourcePathList());

  return Tool.run(
      clang::tooling::newFrontendActionFactory<CRWPluginAction>().get());
}

