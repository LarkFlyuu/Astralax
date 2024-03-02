//===----------------------------------------------------------------------===//
// ShapeInfer Interface
//===----------------------------------------------------------------------===//

#ifndef ASTL_HLD_SHAPEINFER_INTFC
#define ASTL_HLD_SHAPEINFER_INTFC

include "mlir/IR/OpBase.td"

def ShapeInterface : OpInterface<"ShapeInterface"> {
  let description = [{
  }];
  let cppNamespace = "::astl::hld";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          if output shape is dynamic, inference output shape by input shape;
          or verify output shape
        }],
        /*retType=*/"void",
        /*methodName=*/"shape_inference",
        /*args=*/(ins)
      >,
  ];
}
#endif // ASTL_HLD_SHAPEINFER_INTFC
