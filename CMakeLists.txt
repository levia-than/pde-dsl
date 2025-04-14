cmake_minimum_required(VERSION 3.13.4)

project(pde-dsl VERSION 0.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json 便于代码高亮
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(pde-dsl backend/main.cpp)

# 获取需要的LLVM组件
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        IRReader
        Target
        TargetParser
        MC
        X86CodeGen  # 或其他目标架构
        X86AsmParser
        X86Desc
        Analysis
        TransformUtils
)


# Link against LLVM and MLIR libraries
set(LIBS
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRTensorDialect
        MLIRLLVMDialect
        MLIRVectorDialect
        MLIRLLVMIRTransforms
        MLIRLLVMToLLVMIRTranslation
        MLIRLinalgTransforms
        MLIRSCFToControlFlow
        MLIRFuncToLLVM
        MLIRArithToLLVM
        MLIRControlFlowToLLVM
        MLIRMemRefToLLVM
        MLIRPass
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRTransformUtils
        MLIRLinalgTransforms
        MLIRIR
        MLIRParser
        MLIRFuncDialect
        MLIRArithDialect
        MLIRBuiltinToLLVMIRTranslation
        # 方言转换库
        MLIRAMXToLLVMIRTranslation
        MLIRArmSMEToLLVMIRTranslation
        MLIRArmSVEToLLVMIRTranslation
        MLIRArmNeonToLLVMIRTranslation
        MLIRGPUToLLVMIRTranslation
        MLIRNVVMToLLVMIRTranslation
        MLIROpenACCToLLVMIRTranslation
        MLIROpenMPToLLVMIRTranslation
        MLIRROCDLToLLVMIRTranslation
        MLIRSPIRVToLLVMIRTranslation
        MLIRX86VectorToLLVMIRTranslation
        MLIRTensorToLinalg
        MLIRVectorToSCF
        MLIRGPUTransforms
        MLIRReconcileUnrealizedCasts
        # LLVM核心库
)

target_link_libraries(
        pde-dsl
        ${llvm_libs}
        ${LIBS}
)

target_compile_options(pde-dsl PRIVATE
        -fno-rtti
)