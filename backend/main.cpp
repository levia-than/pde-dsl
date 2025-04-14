#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

// Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// Transformations & Conversions
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

// LLVM Translation
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

// LLVM code generation
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CodeGen.h"

#include <optional>


#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"


using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                        llvm::cl::desc("<input file>"),
                                        llvm::cl::Required);

static llvm::cl::opt<std::string> outputFilename("o",
                                         llvm::cl::desc("Output filename"),
                                         llvm::cl::value_desc("filename"),
                                         llvm::cl::init("output.o"));

int main(int argc, char **argv) {
    // Parse command line options
    llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Compiler\n");

    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Set up the MLIR context
    // 设置 MLIR 上下文
    MLIRContext context;

    // 创建方言注册表
    DialectRegistry registry;

    // 注册所有需要的方言到注册表
    registry.insert<
        mlir::func::FuncDialect,
        mlir::LLVM::LLVMDialect,
        mlir::tensor::TensorDialect,
        mlir::linalg::LinalgDialect,
        mlir::scf::SCFDialect,
        mlir::cf::ControlFlowDialect,
        mlir::arith::ArithDialect,
        mlir::memref::MemRefDialect,
        mlir::bufferization::BufferizationDialect,
        mlir::vector::VectorDialect
    >();

    // 注册所有到LLVM IR的转换
    mlir::registerAllToLLVMIRTranslations(registry);

    // 将注册表加载到上下文
    context.appendDialectRegistry(registry);

    // 注册具体的转换接口
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);
    context.enableMultithreading(false);
    context.printOpOnDiagnostic(true);


    // Parse the input file
    OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFilename, &context);
    if (!module) {
        llvm::errs() << "Error: Could not parse input file: " << inputFilename << "\n";
        return 1;
    }

    // Print the original MLIR module
    llvm::outs() << "Original MLIR module:\n";
    module->print(llvm::outs());
    llvm::outs() << "\n\n";

    // Create a pass manager
    PassManager pm(&context);

    mlir::ConversionTarget mlir_target(context);
    mlir_target.addLegalDialect<LLVM::LLVMDialect>();
    mlir_target.addIllegalDialect<func::FuncDialect>();

    // Enable verification for debugging
    pm.enableVerifier(/*verifyPasses=*/true);
    pm.enableStatistics();
    pm.enableIRPrinting();

    // Add conversion passes in the correct order
    {
        OpPassManager &funcPM = pm.nest<func::FuncOp>();
        funcPM.addPass(createLinalgBufferizePass());
        funcPM.addPass(tensor::createTensorBufferizePass());
        funcPM.addPass(createSCFBufferizePass());
        funcPM.addPass(createConvertLinalgToLoopsPass());
        funcPM.addPass(createConvertSCFToCFPass());


    }

    pm.addPass(arith::createArithBufferizePass());
    pm.addPass(mlir::func::createFuncBufferizePass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    // 3. 内存相关转换
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    pm.addPass(createConvertVectorToSCFPass());
    // 4. 转换到 LLVM 方言
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    // pm.addPass(bufferization::createFinalizingBufferizePass());

    // 5. 最后的优化
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    // Run the passes
    if (failed(pm.run(*module))) {
        llvm::errs() << "Error: Failed to lower MLIR to LLVM dialect\n";
        return 1;
    }

    // Print the lowered MLIR module
    llvm::outs() << "Lowered MLIR module (LLVM dialect):\n";
    module->print(llvm::outs());
    llvm::outs() << "\n\n";

    // Translate MLIR LLVM dialect to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Error: Failed to translate MLIR to LLVM IR\n";
        return 1;
    }

    // Print LLVM IR
    llvm::outs() << "Generated LLVM IR:\n";
    llvmModule->print(llvm::outs(), nullptr);
    llvm::outs() << "\n\n";

    // Initialize the target information for code generation
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    llvmModule->setTargetTriple(targetTriple);

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        llvm::errs() << "Error: " << error;
        return 1;
    }

    // 修改这部分代码
    auto CPU = "generic";
    auto features = "";

    auto targetMachine = target->createTargetMachine(targetTriple, CPU, features, {}, {});
    llvmModule->setDataLayout(targetMachine->createDataLayout());

    // 设置输出文件
    std::error_code EC;
    llvm::raw_fd_ostream dest(outputFilename, EC, llvm::sys::fs::OF_None);
    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message() << "\n";
        return 1;
    }

    // 修改文件类型的定义
    // 使用 llvm::CodeGenFileType::CGFT_ObjectFile 替代 llvm::CGFT_ObjectFile
    auto fileType = llvm::CodeGenFileType::ObjectFile;

    // Configure the module and generate code
    llvm::legacy::PassManager pass;

    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
        llvm::errs() << "TargetMachine can't emit a file of this type\n";
        return 1;
    }

    pass.run(*llvmModule);
    dest.flush();

    llvm::outs() << "Generated object file: " << outputFilename << "\n";

    return 0;
}