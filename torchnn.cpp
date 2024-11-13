#include "torchnn.h"
/*
    compress an image
*/
void torchnn::compressJPEG(const std::string& inputFilename, const std::string& outputFilename, int quality) {
    if(inputFilename.empty() || outputFilename.empty()){
        return;
    }
    // Create a JPEG compression struct and error handler
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    // Open the input file
    FILE* infile = fopen(inputFilename.c_str(), "rb");
    if (!infile) {
        throw std::runtime_error("Unable to open input file: " + inputFilename);
    }
    // Initialize JPEG decompression
    jpeg_decompress_struct dinfo;
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);
    jpeg_stdio_src(&dinfo, infile);
    jpeg_read_header(&dinfo, TRUE);
    jpeg_start_decompress(&dinfo);
    // Get image properties
    int width = dinfo.output_width;
    int height = dinfo.output_height;
    int numChannels = dinfo.num_components;
    // Allocate memory for the image data
    unsigned char* buffer = new unsigned char[width * height * numChannels];
    while (dinfo.output_scanline < height) {
        unsigned char* row_pointer = buffer + dinfo.output_scanline * width * numChannels;
        jpeg_read_scanlines(&dinfo, &row_pointer, 1);
    }
    // Finish decompression and close the input file
    jpeg_finish_decompress(&dinfo);
    fclose(infile);
    jpeg_destroy_decompress(&dinfo);
    // Set up compression
    FILE* outfile = fopen(outputFilename.c_str(), "wb");
    if (!outfile) {
        delete[] buffer;
        throw std::runtime_error("Unable to open output file: " + outputFilename);
    }
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = numChannels;
    cinfo.in_color_space = (numChannels == 3) ? JCS_RGB : JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_stdio_dest(&cinfo, outfile);
    // Start compression
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char* row_pointer = buffer + cinfo.next_scanline * width * numChannels;
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    // Finish compression and clean up
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
    delete[] buffer;
    std::cout << "JPEG image compressed and saved to " << outputFilename << std::endl;
}
void torchnn::img_compress(const std::string& input_folder,int quality){
    if(input_folder.empty()){
        return;
    }
     if (!std::filesystem::exists(input_folder)) {
        std::cerr << "The folder does not exist" << std::endl;
        return;
    }
    try {
        for (const auto& entryMainFolder : std::filesystem::directory_iterator(input_folder)) {  
            if (entryMainFolder.is_directory()) {  
                std::string sub_folder_path = entryMainFolder.path().string();
                std::cout << "Start working on folder: " << sub_folder_path << std::endl;
                for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {  
                    if (entrySubFolder.is_regular_file() && 
                        (entrySubFolder.path().extension() == ".JPEG" || entrySubFolder.path().extension() == ".jpeg")) {   
                        std::string imgFilePath = entrySubFolder.path().string(); 
                        compressJPEG(imgFilePath,imgFilePath,quality);
                        std::cout << "Successfully compressed the image: " << imgFilePath << std::endl;
                    }
                }
            }
        }
    }
    catch(std::exception& ex){
        std::cerr << ex.what() << std::endl;
    }
    catch(...){
        std::cerr << "Unknown errors." << std::endl;
    }
    std::cout << "All jobs are done!" << std::endl;
}
void torchnn::serializeImages(int imageSize, const std::string& mainFolderPath, const std::string& outputPath) const{
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file for writing: " << outputPath << std::endl;
        return;
    }
    // Loop through directories
    for (const auto& dirEntry : std::filesystem::directory_iterator(mainFolderPath)) {
        if (std::filesystem::is_directory(dirEntry)) {
            std::string catalog = dirEntry.path().filename().string();
            // Loop through images in the catalog
            for (const auto& imageEntry : std::filesystem::directory_iterator(dirEntry.path())) {
                if (std::filesystem::is_regular_file(imageEntry)) {
                    std::string imagePath = imageEntry.path().string();
                    if (!isValidImageFile(imagePath)) {
                        std::cout << "Not a valid image file: " << imagePath << std::endl;
                        continue;
                    }
                    cv::Mat image = cv::imread(imagePath,cv::IMREAD_COLOR);
                    if (image.empty()) {
                        std::cout << "Failed to load image: " << imagePath << std::endl;
                        continue;
                    }
                    // Resize image
                    cv::resize(image, image, cv::Size(imageSize, imageSize));
                    std::cout << "Saving the image: " << imagePath << std::endl;
                    // Convert to tensor format
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                    image.convertTo(image, CV_32F, 1.0 / 255.0);
                    // Write catalog name length
                    size_t catalogLength = catalog.size();
                    outFile.write(reinterpret_cast<const char*>(&catalogLength), sizeof(size_t));
                    // Write catalog name
                    outFile.write(catalog.c_str(), catalogLength);
                    // Write image size
                    size_t serializedImageSize = image.total() * image.elemSize();
                    outFile.write(reinterpret_cast<const char*>(&serializedImageSize), sizeof(size_t));
                    // Write image data
                    outFile.write(reinterpret_cast<const char*>(image.data), serializedImageSize);
                }
            }
        }
    }
    outFile.close();
}
std::vector<ImageData> torchnn::loadDataset(const std::string& dataset_dat_Path, int imageSize, const std::string& labelMap){
    std::vector<ImageData> dataset;
    std::map<int,std::string> label_mapping;
    if(dataset_dat_Path.empty()){
        return dataset;
    }
    if(std::filesystem::exists(dataset_dat_Path)){
        int currentLabel = 0;
        std::map<std::string,std::vector<cv::Mat>> img_to_train = loadImages(imageSize,dataset_dat_Path);
        if(!img_to_train.empty()){
            for(const auto& item : img_to_train){
                const auto& item_value = item.second;
                for(const auto& img : item_value){
                    auto imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
                    imgTensor = imgTensor.permute({2, 0, 1}); // [H,W,C] -> [C,H,W]
                    imgTensor = imgTensor.clone(); // Ensure that tensor owns its memory
                    imgTensor[0] = imgTensor[0].sub_(0.485).div_(0.229);
                    imgTensor[1] = imgTensor[1].sub_(0.456).div_(0.224);
                    imgTensor[2] = imgTensor[2].sub_(0.406).div_(0.225);
                    dataset.push_back({imgTensor, currentLabel});
                    label_mapping[currentLabel]=item.first;
                }
                currentLabel++;
            }
        }
        else{
            std::cerr << "torchnn::loadDataset img_to_train is empty!" << std::endl;
        }
    }
    if(!label_mapping.empty()){
        std::ofstream ofile(labelMap,std::ios::out);
        if(!ofile.is_open()){
            ofile.open(labelMap,std::ios::out);
        }
        for(const auto& item : label_mapping){
            ofile << item.first << "," << item.second << '\n';
        }
        ofile.close();
    }
    return dataset;
}
void torchnn::train_model(const std::string& dataset_dat_Path,const std::string& modelPath,const std::string& labelMap_path, int imageSize, int numEpochs, int batchSize, float learningRate){
    if(dataset_dat_Path.empty() || modelPath.empty()){
        return;
    }
    // Set device to CPU or CUDA if available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    std::cout << "Start training, please wait..." << std::endl;
    // Parameters
    //const std::string dataset_dat_Path = "/Users/dengfengji/ronnieji/lib/project/main/images.dat"; // Path to your main dataset folder
    //const std::string modelPath = "/Users/dengfengji/ronnieji/lib/project/main/siamese_model.pt";
    // Load dataset
    auto dataset = loadDataset(dataset_dat_Path,imageSize,labelMap_path);
    // Create image pairs and labels
    std::vector<std::pair<torch::Tensor, torch::Tensor>> imagePairs;
    std::vector<int> pairLabels;
    createImagePairs(dataset, imagePairs, pairLabels);
    // Initialize model
    SiameseNetwork model;
    model->to(device);
    // Create Adam optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learningRate));
    // Train the model
    trainModel(model, imagePairs, pairLabels, optimizer, numEpochs, batchSize);
    // Save the model
    torch::save(model, modelPath);
    std::cout << "Successfully saved the model to: " << modelPath << std::endl;
}
std::map<int,std::string> torchnn::read_label_map(const std::string& lable_map_path){
    std::map<int,std::string> label_mapping;
    if(lable_map_path.empty()){
        return label_mapping;
    }
    std::ifstream ifile(lable_map_path);
    if(ifile.is_open()){
        std::string line;
        while(std::getline(ifile, line)){
            if(!line.empty()){
                std::vector<std::string> get_line = JsplitString(line,',');
                if(!get_line.empty()){
                    int label = std::stoi(get_line[0]);
                    std::string className = get_line[1];
                    label_mapping[label] = className;
                }
            }
        }
    }
    ifile.close();
    return label_mapping;
}
void torchnn::initialize_embeddings(
    const std::string& dataset_dat_Path, 
    const std::string& modelPath, 
    const std::string& labelMapPath, 
    SiameseNetwork& model, 
    torch::Device& device, 
    int imageSize, 
    std::vector<std::vector<testImageData>>& oDataSet
) {
    if (dataset_dat_Path.empty() || modelPath.empty() || labelMapPath.empty()) {
        std::cerr << "Error: One or more paths are empty." << std::endl;
        return;
    }
    // Load label mapping file
    std::map<int, std::string> labelMap = read_label_map(labelMapPath);
    if (labelMap.empty()) {
        std::cerr << "Error: Label map could not be loaded." << std::endl;
        return;
    }
    // Load dataset and embeddings
    std::cout << "Loading dataset with embeddings..." << std::endl;
    oDataSet = loadDatasetWithEmbeddings(dataset_dat_Path, labelMapPath, imageSize, model, device);
    if (oDataSet.empty()) {
        std::cerr << "Error: Failed to load dataset with embeddings." << std::endl;
    } else {
        std::cout << "Dataset with embeddings successfully loaded." << std::endl;
    }
}
void torchnn::testModel(const std::string& imagePath, int imageSize, const std::vector<std::vector<testImageData>>& dataset, const std::map<int, std::string>& labelMap, SiameseNetwork& model, torch::Device& device, std::vector<std::string>& strRes, int quality) {
    if (imagePath.empty() || dataset.empty() || labelMap.empty()) {
        return;
    }
    std::vector<std::string> objs_found;
    std::unordered_map<std::string,double> obj_scores;
    // Load and preprocess new image
    torch::Tensor newImageTensor = load_and_preprocess_image(imagePath,imageSize,quality);
    if (newImageTensor.numel() > 0) {
        newImageTensor = newImageTensor.to(device);
        torch::Tensor newImageEmbedding = model->subnetwork->forward(newImageTensor);
        newImageEmbedding = newImageEmbedding.flatten(); // Flatten the embedding
        double similarityThreshold = 0.75; // Similarity threshold
        std::map<int, double> detectedObjects; // Store object labels with their max similarity
        for (const auto& data : dataset) {
            for (const auto& item : data) {
                torch::Tensor datasetEmbedding = item.embedding.to(device);
                double similarity = computeCosineSimilarity(newImageEmbedding, datasetEmbedding);
                // Check if similarity is above the threshold
                if (similarity >= similarityThreshold) {
                    auto it = detectedObjects.find(item.label);
                    if (it == detectedObjects.end() || similarity > it->second) {
                        detectedObjects[item.label] = similarity;
                    }
                }
            }
        }
        // Output all detected object names
        if (!detectedObjects.empty()) {
            //std::cout << imagePath << " contains the following objects:" << std::endl;
            for (const auto& obj : detectedObjects) {
                auto it = labelMap.find(obj.first);
                if (it != labelMap.end()) {
                    if(obj.second == 1.0){
                        objs_found.push_back(it->second);
                    }
                    else if(obj.second > 0.99 && obj.second < 1.0){
                        obj_scores[it->second] = obj.second;
                        //std::cout << "- " << it->second << " (similarity: " << obj.second << ")" << std::endl;
                    }
                }
            }
        } else {
            std::cout << "No significant objects found in " << imagePath << " that match the threshold." << std::endl;
        }
        if(!objs_found.empty()){
            for(const auto& item : objs_found){
                strRes.push_back(item);
                //std::cout << imagePath << " has: " << item << std::endl;
            }
        }
        else{
           if(!obj_scores.empty()){
                std::vector<std::pair<std::string, double>> sorted_score_counting(obj_scores.begin(), obj_scores.end());
                // Sort the vector of pairs
                std::sort(sorted_score_counting.begin(), sorted_score_counting.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });
                auto it = sorted_score_counting.begin();
                //std::cout << imagePath << " is a(an): " << it->first << std::endl;
                strRes.push_back(it->first);
           }
        }
        removeDuplicates(strRes);
    } else {
        std::cerr << "Failed to load new image: " << imagePath << std::endl;
    }
}