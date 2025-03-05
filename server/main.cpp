#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <uWebSockets/App.h>

// Model metadata
const std::vector<std::string> LABELS = {
    "black_belt", "blue_belt", "brown_belt", "gold_belt", "gray_belt",
    "green_belt", "orange_belt", "purple_belt", "red_belt", "resistor",
    "white_belt", "yellow_belt"
};
const int IMG_SIZE = 640; // Input size from metadata: 640x640

// Structure for detection results
struct Detection {
    float box[4]; // [x, y, width, height]
    std::string label;
};

// Base64 decoding function (simplified, assumes valid input)
std::vector<uchar> base64Decode(const std::string& base64) {
    std::string decoded;
    const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int padding = 0;
    for (size_t i = 0; i < base64.length(); i += 4) {
        int a = base64_chars.find(base64[i]);
        int b = base64_chars.find(base64[i + 1]);
        int c = base64_chars.find(base64[i + 2]);
        int d = base64_chars.find(base64[i + 3]);
        decoded.push_back((a << 2) | (b >> 4));
        if (base64[i + 2] != '=') decoded.push_back(((b & 15) << 4) | (c >> 2));
        if (base64[i + 3] != '=') decoded.push_back(((c & 3) << 6) | d);
        if (base64[i + 2] == '=') padding++;
        if (base64[i + 3] == '=') padding++;
    }
    return std::vector<uchar>(decoded.begin(), decoded.end());
}

// Load and preprocess image
cv::Mat preprocessImage(const std::string& base64Data) {
    // Decode base64 to binary
    std::string imageData = base64Data.substr(base64Data.find(",") + 1); // Skip "data:image/jpeg;base64,"
    std::vector<uchar> decoded = base64Decode(imageData);

    // Convert to OpenCV Mat
    cv::Mat img = cv::imdecode(decoded, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to decode image" << std::endl;
        return cv::Mat();
    }

    // Resize to 640x640
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(IMG_SIZE, IMG_SIZE));

    // Convert BGR to RGB and normalize to [0, 1]
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    return resized;
}

// Post-process model output
std::vector<Detection> postProcess(tflite::Interpreter* interpreter, float threshold = 0.5) {
    std::vector<Detection> detections;

    // Assuming output tensor is [1, num_detections, 6] (x, y, w, h, confidence, class)
    auto* output = interpreter->typed_output_tensor<float>(0);
    int num_detections = interpreter->outputs()[0].dims->data[1]; // Adjust based on model output shape

    for (int i = 0; i < num_detections; ++i) {
        float confidence = output[i * 6 + 4];
        if (confidence > threshold) {
            float x = output[i * 6 + 0];
            float y = output[i * 6 + 1];
            float w = output[i * 6 + 2];
            float h = output[i * 6 + 3];
            int class_id = static_cast<int>(output[i * 6 + 5]);

            Detection det;
            det.box[0] = x - w / 2; // Convert center to top-left
            det.box[1] = y - h / 2;
            det.box[2] = w;
            det.box[3] = h;
            det.label = LABELS[class_id];
            detections.push_back(det);
        }
    }
    return detections;
}

int main() {
    // Load TF Lite model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("best_float32.tflite");
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter" << std::endl;
        return -1;
    }

    interpreter->AllocateTensors();

    // Initialize uWebSockets server
    uWS::App app;
    app.ws("/*", {
        .open = [](auto* ws) {
            std::cout << "Client connected" << std::endl;
        },
        .message = [&](auto* ws, std::string_view message, uWS::OpCode opCode) {
            // Parse JSON message from client
            std::string msg(message);
            size_t idPos = msg.find("\"id\":");
            size_t dataPos = msg.find("\"data\":\"") + 8;
            size_t dataEnd = msg.rfind("\"");
            if (idPos == std::string::npos || dataPos == std::string::npos || dataEnd == std::string::npos) {
                std::cerr << "Invalid message format" << std::endl;
                return;
            }

            int frameId = std::stoi(msg.substr(idPos + 5, dataPos - idPos - 8));
            std::string base64Data = msg.substr(dataPos, dataEnd - dataPos);

            // Preprocess image
            cv::Mat inputImg = preprocessImage(base64Data);
            if (inputImg.empty()) return;

            // Copy to TF Lite input tensor
            float* input = interpreter->typed_input_tensor<float>(0);
            memcpy(input, inputImg.data, IMG_SIZE * IMG_SIZE * 3 * sizeof(float));

            // Run inference
            interpreter->Invoke();

            // Post-process results
            std::vector<Detection> detections = postProcess(interpreter);

            // Prepare JSON response
            std::string response = "{\"id\":" + std::to_string(frameId) + ",\"detections\":[";
            for (size_t i = 0; i < detections.size(); ++i) {
                auto& det = detections[i];
                response += "{\"box\":[" + std::to_string(det.box[0]) + "," +
                            std::to_string(det.box[1]) + "," +
                            std::to_string(det.box[2]) + "," +
                            std::to_string(det.box[3]) + "],\"label\":\"" + det.label + "\"}";
                if (i < detections.size() - 1) response += ",";
            }
            response += "]}";

            // Send response back to client
            ws->send(response, uWS::OpCode::TEXT);
        },
        .close = [](auto* ws, int code, std::string_view message) {
            std::cout << "Client disconnected" << std::endl;
        }
    }).listen(5000, [](auto* listen_socket) {
        if (listen_socket) {
            std::cout << "Server listening on port 5000" << std::endl;
        } else {
            std::cerr << "Failed to listen on port 5000" << std::endl;
        }
    }).run();

    return 0;
}
