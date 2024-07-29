//==============================================================================
// FILE:
//    CudaRW.h
//
// DESCRIPTION:
//
// License: The Unlicense
//==============================================================================
#ifndef CUDARW_H
#define CUDARW_H

#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Rewrite/Core/Rewriter.h"

//-----------------------------------------------------------------------------
// ASTMatcher callback
//-----------------------------------------------------------------------------
class CudaRWMatcher
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  CudaRWMatcher(clang::Rewriter &CRWRewriter) : CRWRewriter(CRWRewriter) {}
  // Callback that's executed whenever the Matcher in CudaRWASTConsumer
  // matches.
  void run(const clang::ast_matchers::MatchFinder::MatchResult &) override;
  // Callback that's executed at the end of the translation unit
  void onEndOfTranslationUnit() override;

private:
  clang::Rewriter CRWRewriter;
  llvm::SmallSet<clang::FullSourceLoc, 8> EditedLocations;
};

//-----------------------------------------------------------------------------
// ASTConsumer
//-----------------------------------------------------------------------------
class CudaRWASTConsumer : public clang::ASTConsumer {
public:
  CudaRWASTConsumer(clang::Rewriter &R);
  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    Finder.matchAST(Ctx);
  }

private:
  clang::ast_matchers::MatchFinder Finder;
  CudaRWMatcher CRWHandler;
};

#endif

