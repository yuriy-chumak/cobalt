/*
 * Copyright 2020 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "src/sksl/SkSLThreadContext.h"

#include "include/private/SkSLProgramElement.h"
#include "include/sksl/DSLSymbols.h"
#include "include/sksl/SkSLPosition.h"
#include "src/sksl/SkSLBuiltinMap.h"
#include "src/sksl/SkSLCompiler.h"
#include "src/sksl/SkSLModifiersPool.h"
#include "src/sksl/SkSLParsedModule.h"
#include "src/sksl/SkSLPool.h"
#include "src/sksl/SkSLUtil.h"
#include "src/sksl/ir/SkSLExternalFunction.h"
#include "src/sksl/ir/SkSLSymbolTable.h"

#include <type_traits>

#if !defined(SKSL_STANDALONE) && SK_SUPPORT_GPU
#include "src/gpu/ganesh/glsl/GrGLSLFragmentShaderBuilder.h"
#endif // !defined(SKSL_STANDALONE) && SK_SUPPORT_GPU

#if defined(STARBOARD)
#include <pthread.h>

#include "starboard/common/log.h"
#include "starboard/thread.h"
#endif

namespace SkSL {

ThreadContext::ThreadContext(SkSL::Compiler* compiler, SkSL::ProgramKind kind,
        const SkSL::ProgramSettings& settings, SkSL::ParsedModule module, bool isModule)
    : fCompiler(compiler)
    , fOldErrorReporter(*fCompiler->fContext->fErrors)
    , fSettings(settings) {
    fOldModifiersPool = fCompiler->fContext->fModifiersPool;

    fOldConfig = fCompiler->fContext->fConfig;

    if (!isModule) {
        if (compiler->context().fCaps.useNodePools() && settings.fDSLUseMemoryPool) {
            fPool = Pool::Create();
            fPool->attachToThread();
        }
        fModifiersPool = std::make_unique<SkSL::ModifiersPool>();
        fCompiler->fContext->fModifiersPool = fModifiersPool.get();
    }

    fConfig = std::make_unique<SkSL::ProgramConfig>();
    fConfig->fKind = kind;
    fConfig->fSettings = settings;
    fConfig->fIsBuiltinCode = isModule;
    fCompiler->fContext->fConfig = fConfig.get();
    fCompiler->fContext->fErrors = &fDefaultErrorReporter;
    fCompiler->fContext->fBuiltins = module.fElements.get();
    if (fCompiler->fContext->fBuiltins) {
        fCompiler->fContext->fBuiltins->resetAlreadyIncluded();
    }

    fCompiler->fSymbolTable = module.fSymbols;
    this->setupSymbolTable();
}

ThreadContext::~ThreadContext() {
    if (SymbolTable()) {
        fCompiler->fSymbolTable = nullptr;
        fProgramElements.clear();
    } else {
        // We should only be here with a null symbol table if ReleaseProgram was called
        SkASSERT(fProgramElements.empty());
    }
    fCompiler->fContext->fErrors = &fOldErrorReporter;
    fCompiler->fContext->fConfig = fOldConfig;
    fCompiler->fContext->fModifiersPool = fOldModifiersPool;
    if (fPool) {
        fPool->detachFromThread();
    }
}

void ThreadContext::setupSymbolTable() {
    SkSL::Context& context = *fCompiler->fContext;
    SymbolTable::Push(&fCompiler->fSymbolTable, context.fConfig->fIsBuiltinCode);

    if (fSettings.fExternalFunctions) {
        // Add any external values to the new symbol table, so they're only visible to this Program.
        SkSL::SymbolTable& symbols = *fCompiler->fSymbolTable;
        for (const std::unique_ptr<ExternalFunction>& ef : *fSettings.fExternalFunctions) {
            symbols.addWithoutOwnership(ef.get());
        }
    }
}

SkSL::Context& ThreadContext::Context() {
    return Compiler().context();
}

const SkSL::ProgramSettings& ThreadContext::Settings() {
    return Context().fConfig->fSettings;
}

std::shared_ptr<SkSL::SymbolTable>& ThreadContext::SymbolTable() {
    return Compiler().fSymbolTable;
}

const SkSL::Modifiers* ThreadContext::Modifiers(const SkSL::Modifiers& modifiers) {
    return Context().fModifiersPool->add(modifiers);
}

ThreadContext::RTAdjustData& ThreadContext::RTAdjustState() {
    return Instance().fRTAdjust;
}

#if !defined(SKSL_STANDALONE) && SK_SUPPORT_GPU
void ThreadContext::StartFragmentProcessor(GrFragmentProcessor::ProgramImpl* processor,
        GrFragmentProcessor::ProgramImpl::EmitArgs* emitArgs) {
    ThreadContext& instance = ThreadContext::Instance();
    instance.fStack.push({processor, emitArgs, StatementArray{}});
    CurrentEmitArgs()->fFragBuilder->fDeclarations.swap(instance.fStack.top().fSavedDeclarations);
    dsl::PushSymbolTable();
}

void ThreadContext::EndFragmentProcessor() {
    ThreadContext& instance = Instance();
    SkASSERT(!instance.fStack.empty());
    CurrentEmitArgs()->fFragBuilder->fDeclarations.swap(instance.fStack.top().fSavedDeclarations);
    instance.fStack.pop();
    dsl::PopSymbolTable();
}
#endif // !defined(SKSL_STANDALONE) && SK_SUPPORT_GPU

void ThreadContext::SetErrorReporter(ErrorReporter* errorReporter) {
    SkASSERT(errorReporter);
    Context().fErrors = errorReporter;
}

void ThreadContext::ReportError(std::string_view msg, Position pos) {
    GetErrorReporter().error(msg, pos);
}

void ThreadContext::DefaultErrorReporter::handleError(std::string_view msg, Position pos) {
    SK_ABORT("error: %.*s\nNo SkSL error reporter configured, treating this as a fatal error\n",
             (int)msg.length(), msg.data());
}

void ThreadContext::ReportErrors(Position pos) {
    GetErrorReporter().reportPendingErrors(pos);
}

#if !defined(STARBOARD)
thread_local ThreadContext* instance = nullptr;

bool ThreadContext::IsActive() {
    return instance != nullptr;
}

ThreadContext& ThreadContext::Instance() {
    SkASSERTF(instance, "dsl::Start() has not been called");
    return *instance;
}

void ThreadContext::SetInstance(std::unique_ptr<ThreadContext> newInstance) {
    SkASSERT((instance == nullptr) != (newInstance == nullptr));
    delete instance;
    instance = newInstance.release();
}
#else
namespace {
ThreadContext* instance = nullptr;

pthread_once_t s_once_flag = PTHREAD_ONCE_INIT;
SbThreadLocalKey s_thread_local_key = kSbThreadLocalKeyInvalid;

void InitThreadLocalKey() {
    s_thread_local_key = SbThreadCreateLocalKey(nullptr);
    SB_DCHECK(SbThreadIsValidLocalKey(s_thread_local_key));
    SbThreadSetLocalValue(s_thread_local_key, nullptr);
}

void EnsureThreadLocalKeyInited() {
    pthread_once(&s_once_flag, InitThreadLocalKey);
    SB_DCHECK(SbThreadIsValidLocalKey(s_thread_local_key));
}
}  // namespace

bool ThreadContext::IsActive() { return SbThreadGetLocalValue(s_thread_local_key) != nullptr; }

ThreadContext& ThreadContext::Instance() { return *instance; }

void ThreadContext::SetInstance(std::unique_ptr<ThreadContext> newInstance) {
    EnsureThreadLocalKeyInited();
    delete instance;
    SbThreadSetLocalValue(s_thread_local_key, static_cast<void*>(newInstance.release()));
    instance = static_cast<ThreadContext*>(SbThreadGetLocalValue(s_thread_local_key));
}

#endif

} // namespace SkSL
