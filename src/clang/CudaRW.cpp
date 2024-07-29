//==============================================================================
// FILE:
//    CudaRW.cpp
//
// DESCRIPTION: TODO replace this
//    Literal argument commenter - adds in-line C-style comments as recommended
//    in LLVM's coding guideline:
//      * https://llvm.org/docs/CodingStandards.html#comment-formatting
//    This plugin will comment the following literal argument types:
//      * integer, character, floating, boolean, string
//
//    Below is an example for a function that takes one integer argument:
//    ```c
//    extern void foo(int some_arg);
//
//    void(bar) {
//      foo(/*some_arg=*/123);
//    }
//    ```
//
// USAGE:
//    <BUILD_DIR>/bin/ct-la-commenter test/CRWInt.cpp
//
// REFERENCES:
//    Based on an example by Peter Smith:
//      * https://s3.amazonaws.com/connect.linaro.org/yvr18/presentations/yvr18-223.pdf
//
// License: The Unlicense
//==============================================================================
//
//
//
//
//
//
//
//==============================================================================
//
// Strategy:
//  1. update matcher
//    i. bind relevant sites (@ StatementMatcher)
//      - driver:
//        + copy device data structures (infer from cuda malloc?) for metadata
//        + change kernel call to include metadata arrays
//        + add verification printout at end
//      - kernel:
//        + copy global data structure params in kernel decl
//        + create barrier counter
//        + identify any new global memory instantiation
//        + create offsets for global memory aliases
//        + instrument log calls for all reads & writes to global memory
//    ii. what keywords are necessary?
//  2. update matcher::run
//    - 
//
//
//
//==============================================================================
#include "CudaRW.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ast_matchers;

//-----------------------------------------------------------------------------
// CudaRW - implementation
//-----------------------------------------------------------------------------
void CudaRWMatcher::run(const MatchFinder::MatchResult &Result) {
  // ASTContext is used to retrieve the source location
  ASTContext *Ctx = Result.Context;

  const FunctionDecl *KernelDecl = 
    Result.Nodes.getNodeAs<clang::FunctionDecl>("kernel");
  
  const CallExpr *SyncCall = 
    Result.Nodes.getNodeAs<clang::CallExpr>("synccall");
  
  // Basic sanity checking
  // assert(KernelDecl && SyncCall &&
  //        "The matcher matched, so nodes should be non-null");
 
  
  /*
   * Handle kernel processing
   */
  if(KernelDecl) {
    const FunctionDecl *KernelDef = nullptr;
    if (!KernelDecl->hasBody(KernelDef))
      return;
    
    // Add log params
    if (!KernelDef->param_empty()) {
      unsigned numParams = KernelDef->getNumParams();

      // For each argument match it with the callee parameter. If it is an integer,
      // float, boolean, character or string literal insert a comment.
      for (unsigned i = 0; i < numParams; i++) {
        ParmVarDecl *ParamDecl = KernelDef->parameters()[i];

        if (ParamDecl->getDeclName().getAsString() == "data1" ||
            ParamDecl->getDeclName().getAsString() == "data2") {
          CRWRewriter.ReplaceText(ParamDecl->getTypeSourceInfo()->getTypeLoc().getSourceRange(), "data_ptr");
        }
      }
      
      
      // Add metadata args
      FullSourceLoc KernParamLoc = Ctx->getFullLoc(KernelDef->parameters()[numParams-1]->getEndLoc().getLocWithOffset(4));
      
      std::string newParams = ", uint64_cu* metadata1, uint64_cu* metadata2"; 
      CRWRewriter.InsertText(KernParamLoc, newParams);
    }
    
    // add inits to kernel
    const Stmt *KernelBody = KernelDef->getBody();
    
    FullSourceLoc KernStartLoc = Ctx->getFullLoc(KernelBody->getBeginLoc().getLocWithOffset(1));
    
    std::string initialSetup = 
      "\n  /*** CRW Begin ***/\n  "
      "unsigned bcount;\n  "
      "data1.setMetadata(metadata1);\n  "
      "data2.setMetadata(metadata2);\n  "
      "/*** CRW End ***/\n";
      
    CRWRewriter.InsertText(KernStartLoc, initialSetup);
  }
  
  
  /*
   * Handle __syncthreads calls
   */
  if (SyncCall) {
    FullSourceLoc SyncCallLoc = Ctx->getFullLoc(SyncCall->getEndLoc().getLocWithOffset(2));
    CRWRewriter.InsertText(SyncCallLoc, " data1.incBcount(); data2.incBcount();");
  }
  // // Callee and caller are accessed via .bind("callee") and .bind("caller"),
  // // respectively, from the ASTMatcher
  // const FunctionDecl *CalleeDecl =
  //     Result.Nodes.getNodeAs<clang::FunctionDecl>("callee");
  // const CallExpr *TheCall = Result.Nodes.getNodeAs<clang::CallExpr>("caller");

  // // Basic sanity checking
  // assert(TheCall && CalleeDecl &&
  //        "The matcher matched, so callee and caller should be non-null");

  // // No arguments means there's nothing to comment
  // if (CalleeDecl->parameters().empty())
  //   return;

  // // Get the arguments
  // Expr const *const *Args = TheCall->getArgs();
  // size_t NumArgs = TheCall->getNumArgs();

  // // If this is a call to an overloaded operator (e.g. `+`), then the first
  // // parameter is the object itself (i.e. `this` pointer). Skip it.
  // if (isa<CXXOperatorCallExpr>(TheCall)) {
  //   Args++;
  //   NumArgs--;
  // }

  // // For each argument match it with the callee parameter. If it is an integer,
  // // float, boolean, character or string literal insert a comment.
  // for (unsigned Idx = 0; Idx < NumArgs; Idx++) {
  //   const Expr *AE = Args[Idx]->IgnoreParenCasts();

  //   // if (!dyn_cast<IntegerLiteral>(AE) && !dyn_cast<CXXBoolLiteralExpr>(AE) &&
  //   //     !dyn_cast<FloatingLiteral>(AE) && !dyn_cast<StringLiteral>(AE) &&
  //   //     !dyn_cast<CharacterLiteral>(AE))
  //   //   continue;

  //   // Parameter declaration
  //   ParmVarDecl *ParamDecl = CalleeDecl->parameters()[Idx];

  //   // Source code locations (parameter and argument)
  //   FullSourceLoc ParamLocation = Ctx->getFullLoc(ParamDecl->getBeginLoc());
  //   FullSourceLoc ArgLoc = Ctx->getFullLoc(AE->getBeginLoc());

  //   if (ParamLocation.isValid() && !ParamDecl->getDeclName().isEmpty() &&
  //       EditedLocations.insert(ArgLoc).second)
  //     // Insert the comment immediately before the argument
  //     CRWRewriter.InsertText(
  //         ArgLoc,
  //         (Twine("metadata_") + ParamDecl->getDeclName().getAsString() + ", ").str());
  // }
}

void CudaRWMatcher::onEndOfTranslationUnit() {
  // Replace in place
  // CRWRewriter.overwriteChangedFiles();

  // Output to stdout
  CRWRewriter.getEditBuffer(CRWRewriter.getSourceMgr().getMainFileID())
      .write(llvm::outs());
}

CudaRWASTConsumer::CudaRWASTConsumer(Rewriter &R) : CRWHandler(R) {
  // StatementMatcher CallSiteMatcher =
  //     callExpr(
  //         allOf(callee(functionDecl(unless(isVariadic())).bind("callee")),
  //               unless(cxxMemberCallExpr(
  //                   on(hasType(substTemplateTypeParmType())))),
  //               anyOf(hasAnyArgument(ignoringParenCasts(cxxBoolLiteral())),
  //                     hasAnyArgument(ignoringParenCasts(integerLiteral())),
  //                     hasAnyArgument(ignoringParenCasts(stringLiteral())),
  //                     hasAnyArgument(ignoringParenCasts(characterLiteral())),
  //                     hasAnyArgument(ignoringParenCasts(floatLiteral())))))
  //         .bind("caller");

  // Match kernel call and sequential , specifically data_t types for now
  // in Indigo, these are specifically the data arrays we need
  // should be generalized later.
  // StatementMatcher CallSiteMatcher =
  //     callExpr(
  //         allOf(callee(functionDecl(unless(isVariadic())).bind("callee")),
  //               unless(cxxMemberCallExpr(
  //                   on(hasType(substTemplateTypeParmType())))),
  //               anyOf(hasAnyArgument(hasType(asString("data_t"))))))
  //         .bind("caller");
  
  DeclarationMatcher KernelDeclMatcher =
    functionDecl(anyOf(hasAttr(clang::attr::CUDAGlobal),
                       hasAttr(clang::attr::CUDADevice)),
                 isDefinition(),
                 unless(isExpansionInSystemHeader())).bind("kernel");
    // functionDecl(anyOf(hasAttr(clang::attr::CUDAGlobal), hasAttr(clang::attr::CUDADevice)),
    //   unless(hasAttr(clang::attr::CUDAHost))).bind("kernel");

  // needed before setting to overwrite
  // isExpansionInMainFile()
  StatementMatcher SyncCallMatcher = 
    callExpr(callee(functionDecl(hasName("__syncthreads")))).bind("synccall");
    // callExpr(callee(functionDecl(matchesName(".*__syncthreads")))).bind("synccall");

  // CRW is the callback that will run when the ASTMatcher finds the pattern
  // above.
  // Finder.addMatcher(CallSiteMatcher, &CRWHandler);
  // Traverse mode set here should ignore messy things i don't care about, hopefully
  Finder.addMatcher(traverse(TK_IgnoreUnlessSpelledInSource, KernelDeclMatcher), &CRWHandler);
  Finder.addMatcher(SyncCallMatcher, &CRWHandler);
}

//-----------------------------------------------------------------------------
// FrontendAction
//-----------------------------------------------------------------------------
class CRWPluginAction : public PluginASTAction {
public:
  // Our plugin can alter behavior based on the command line options
  bool ParseArgs(const CompilerInstance &,
                 const std::vector<std::string> &) override {
    return true;
  }

  // Returns our ASTConsumer per translation unit.
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    RewriterForCRW.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<CudaRWASTConsumer>(RewriterForCRW);
  }

private:
  Rewriter RewriterForCRW;
};

//-----------------------------------------------------------------------------
// Registration
//-----------------------------------------------------------------------------
static FrontendPluginRegistry::Add<CRWPluginAction>
    X(/*Name=*/"CRW",
      /*Desc=*/"Cuda rewriter for instrumenting race detection");

