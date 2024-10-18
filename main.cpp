#include <iostream>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

#define IMG_PATH "256px-Albert_Einstein_Head.jpeg"

int saveImage(int rows, int cols, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> image, string output_image_path){
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image(rows, cols);
    output_image = image.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });

  // Save the image using stb_image_write
  if (stbi_write_png(output_image_path.c_str(), cols, rows, 1, output_image.data(), cols) == 0) {
    cerr << "Error: Could not save inputMatrixscale image" << endl;
    return 1;
  }
  return 0;
}

void exportToMatrixMarket(const Eigen::MatrixXd& mat, const std::string& filename) {
    // Verifica che la matrice non sia vuota
    if (mat.rows() == 0 || mat.cols() == 0) {
        cerr << "Error: Matrix is empty, cannot export to Matrix Market format." << endl;
        return;
    }

    std::ofstream file(filename);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write the Matrix Market header
    file << "%%MatrixMarket matrix coordinate real general\n";
    
    // Scrivi le dimensioni della matrice e il numero di non-zero
    int nonZeros = 0;
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            if (mat(i, j) != 0.0) {
                ++nonZeros;
            }
        }
    }

    file << mat.rows() << " " << mat.cols() << " " << nonZeros << "\n";

    // Loop through elements of the dense matrix
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            if (mat(i, j) != 0.0) {
                // MatrixMarket format uses 1-based indexing, so add 1 to row and col indices
                file << (i + 1) << " " << (j + 1) << " " << mat(i, j) << "\n";
            }
        }
    }

    // Close the file
    file.close();
    std::cout << "Matrix exported successfully to " << filename << std::endl;
}



Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> loadImage(int* ptr_cols, int* ptr_rows,int* ptr_channels){

    // Load the image using stb_image
    // for greyscale images force to load only one channel
   
    unsigned char* image_data;
    image_data = stbi_load(IMG_PATH, ptr_cols, ptr_rows, ptr_channels, 1);
    if (!image_data) {
        cerr << "Error: Could not load image " << IMG_PATH << endl;
        return Matrix<double, Dynamic, Dynamic, RowMajor>();  // Return empty matrix if error occurs.  Replace this with your own error handling
    }
    int rows = *ptr_rows;
    int cols = *ptr_cols;
    int channels = *ptr_channels;
    cout << "Image loaded: " << rows << "x" << cols << " with " << channels << " channels." << endl;
    Matrix<double, Dynamic, Dynamic, RowMajor> inputMatrix(rows, cols);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            int index = (i * cols + j) * 1;
            inputMatrix(i,j) = static_cast<double>(image_data[index]);
        }
    }
    stbi_image_free(image_data);
    return inputMatrix;
}


int computeSingularValues(Matrix<double, Dynamic, Dynamic, RowMajor> A){

    // Perform Singular Value Decomposition (SVD)
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    // Get the singular values
    VectorXd singularValues = svd.singularValues();
    // Output the two largest singular values
    cout << "The two largest singular values of matrix A are:" << endl;
    cout << singularValues(0) << endl;  // Largest singular value of A
    cout << singularValues(1) << endl;  // Second largest singular value of A

    return pow(singularValues(0),2); // Largest eigenvalue of ATA

}


string readFile(string filename, string word_to_find){
    ifstream file(filename);       // Apri il file in modalità lettura

    if (!file.is_open()) {
        return ("Errore nell'apertura del file!");
    }

    string line;
    int line_number = 0;
    bool found = false;

    // Leggi il file riga per riga
    while (getline(file, line)) {
        line_number++;
        // Cerca la parola nella riga corrente
        if (line.find(word_to_find) != string::npos) {
            //cout << "Parola trovata alla riga " << line_number << ": " << line << std::endl;
            found = true;
            return line;
        }
    }

    if (!found) {
        return ("Parola non trovata nel file.");
    }
    file.close();
    return "";
}


Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> createCheckerBoard(){
    int dim = 200;
    int squareDims = 25;
    bool isBlack = true;
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(dim, dim);
    for (int i = 0; i < dim; i++) {
        if (i % squareDims == 0){
            isBlack = !isBlack;
        }
        for (int j = 0; j < dim; j++){
            if (j % squareDims == 0){
                isBlack = !isBlack;
            }
            if (isBlack){
                A(i, j) = 0;
            } else {
                A(i, j) = 255;
            }
        }
    }
    return A;
}





// // CASO 2: Risolvi il problema degli autovalori
// EigenSolver<MatrixXd> es(ATA);
// // Stampa gli autovalori
// VectorXcd eigenvalues = es.eigenvalues();
// cout << "Gli autovalori più grandi di ATA sono:" << endl;
// cout << eigenvalues(0) << endl;
// cout << eigenvalues(1) << endl;
// cout << "Le radici degli autovalori più grandi di ATA sono:" << endl;
// cout << sqrt(abs(eigenvalues(0))) << endl;
// cout << sqrt(abs(eigenvalues(1))) << endl;


void getCols(JacobiSVD<MatrixXd> svd, MatrixXd S, MatrixXd* C1, MatrixXd* D1, MatrixXd* C2, MatrixXd* D2, int k1, int k2){
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::MatrixXd SV = (S * (V.transpose())).transpose();

    *C1 = U.leftCols(k1);
    *D1 = SV.leftCols(k1);

    *C2 = U.leftCols(k2);
    *D2 = SV.leftCols(k2);
}

void task1(Matrix<double, Dynamic, Dynamic, RowMajor> A, Matrix<double, Dynamic, Dynamic, RowMajor> *ATA){
    cout << "---------TASK1---------" << endl<< endl;

    MatrixXd traspose = A.transpose();
    *ATA = traspose*A;
    cout << "The norm of A'A is: " << ATA->norm() << endl;
}

void task2(Matrix<double, Dynamic, Dynamic, RowMajor> A){
    cout << "---------TASK2---------" << endl<< endl;
    
    int lambda1 = computeSingularValues(A);
    cout << "lambda1: " << lambda1 << endl;
}

void task3(Matrix<double, Dynamic, Dynamic, RowMajor> ATA){
    cout << "---------TASK3---------" << endl<< endl;
    
    Eigen::saveMarket(ATA,"matrix_ATA.mtx");

    system("mpirun -n 4 ./lis-2.1.6/test/etest1 matrix_ATA.mtx -e 1 -etol 1.0e-8 > maxEigenLisOutput.txt");

    string maxEigenValue = readFile("maxEigenLisOutput.txt", "eigenvalue");
    cout << maxEigenValue << endl;

}

void task4(){
    //Here starts task 4
    cout << "---------TASK4---------" << endl<< endl;
    for(int shifting=15000000;shifting<60000000;shifting+=2500000){//the shift is valuable(it accellerates the process) when it is >15000000 and <60000000
        string command = ("mpirun -n 4 ./lis-2.1.6/test/etest1 matrix_ATA.mtx -e 1 -etol 1.0e-8 -shift " + to_string(shifting) + " > maxEigenLisOutput.txt");
    
        // Passa la stringa come C-style string utilizzando c_str()
        system(command.c_str());

        // Leggi il valore dell'autovalore massimo
        string maxEigenValue = readFile("maxEigenLisOutput.txt", "number of iterations");
        cout << maxEigenValue << endl;
    }
}

void task5(Matrix<double, Dynamic, Dynamic, RowMajor> A, JacobiSVD<MatrixXd>svd, MatrixXd *S){
    cout << "---------TASK5---------" << endl<< endl;
    Eigen::VectorXd singularValues = svd.singularValues();
    int sizeS = singularValues.size();
    *S = MatrixXd::Zero(sizeS, sizeS);
    for (int i = 0; i < sizeS; ++i) {
        (*S)(i, i) = singularValues(i);
    }
    cout << "Euclidian norm of S as vector:" << singularValues.norm() << endl;
    cout << "Euclidian norm of S as matrix:" << (*S).norm() << endl;

    cout << "ANSWER: Real euclidian norm of S:" << singularValues(0) << endl;

}

void task6_7(JacobiSVD<MatrixXd>svd, MatrixXd S){

    cout << "---------TASK6---------" << endl<< endl;

    Eigen::MatrixXd C1; Eigen::MatrixXd D1; Eigen::MatrixXd C2; Eigen::MatrixXd D2;
    getCols(svd, S, &C1, &D1, &C2, &D2, 40, 80);

    cout << "Non-zero C1:" << C1.nonZeros() << endl;
    cout << "Non-zero D1:" << D1.nonZeros() << endl;
    cout << "Non-zero C2:" << C2.nonZeros() << endl;
    cout << "Non-zero D2:" << D2.nonZeros() << endl;

    cout << "---------TASK7---------" << endl<< endl;

    MatrixXd CD1 = C1 * D1.transpose();
    MatrixXd CD2 = C2 * D2.transpose();
    CD1 = CD1.unaryExpr([](double el) -> double { 
        if (el > 255) el = 255;
        if (el < 0) el = 0;
        return el;
    });
    CD2 = CD2.unaryExpr([](double el) -> double { 
        if (el > 255) el = 255;
        if (el < 0) el = 0;
        return el;
    });
    


    saveImage(CD1.rows(), CD1.cols(), CD1, "CDTranspose_40.png");
    saveImage(CD2.rows(), CD2.cols(), CD2, "CDTranspose_80.png");
}

void task8(int *dim, Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* checkerBoard){
    //Here starts task 8
    cout << "---------TASK8---------" << endl<< endl;

    *dim = 200; 
    *checkerBoard = createCheckerBoard();
    saveImage(*dim, *dim,*checkerBoard,"scacchiera.png");
    cout << "Norm of the checkerboard:" << checkerBoard->norm()<< endl;
}

void task9(int dim, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> *noised, Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> checkerBoard){

    //Here starts task 9
    cout << "---------TASK9---------" << endl<< endl;
    

    Eigen::MatrixXd randomMatrix = Eigen::MatrixXd::Random(dim, dim);
    randomMatrix = 50*randomMatrix;
    // Fill the matrices with image data
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            (*noised)(i, j) = checkerBoard(i,j) + randomMatrix(i, j);
            if ((*noised)(i,j) >= 255) (*noised)(i, j) = 255;
            if ((*noised)(i,j) <= 0) (*noised)(i, j) = 0;
        }
    }

    saveImage(dim, dim, *noised, "noised_task9.png");
}

void task10(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> noised){

    //Here starts task 10
    cout << "---------TASK10---------" << endl;

    computeSingularValues(noised);
}

int main(){
    int cols, rows, channels;    
    Matrix<double, Dynamic, Dynamic, RowMajor> A = loadImage(&cols, &rows, &channels);
    Matrix<double, Dynamic, Dynamic, RowMajor> ATA = loadImage(&cols, &rows, &channels);
    MatrixXd S;
    int dim;
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> checkerBoard;


    task1(A, &ATA);
    task2(A);
    task3(ATA);
    task4();
    JacobiSVD<MatrixXd>svd (A, ComputeThinU | ComputeThinV);
    task5(A, svd, &S);
    task6_7(svd, S);
    task8(&dim, &checkerBoard);
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> noised(dim, dim);
    task9(dim, &noised, checkerBoard);
    task10(noised);
    //task11()









    //Here starts task 11
    cout << "---------TASK11---------" << endl;

    JacobiSVD<MatrixXd> svd2(noised, ComputeThinU | ComputeThinV);
    Eigen::MatrixXd C1; Eigen::MatrixXd D1; Eigen::MatrixXd C2; Eigen::MatrixXd D2;
    getCols(svd, S, &C1, &D1, &C2, &D2, 5, 10);
    //cout << C1.rows() << " " << C1.cols() << " ";



    //Here starts task 12
    cout << "---------TASK12---------" << endl;




    //Here starts task 13
    cout << "---------TASK13---------" << endl;
}

