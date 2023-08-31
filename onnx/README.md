# onnx 前端

onnx-home 是 onnx 主项目，需要里面的 [onnx.proto](onnx-home/onnx/onnx.proto)/[onnx.proto3](onnx-home/onnx/onnx.proto3) 提供的定义来解析 onnx 文件。要编译 protobuf，需要 protoc 编译器，暂时将编译好的 binary 引入项目，目前加入了 win64 和 linux x86-64 两种，版本采用 [24.2](https://github.com/protocolbuffers/protobuf/releases/tag/v24.2)。
