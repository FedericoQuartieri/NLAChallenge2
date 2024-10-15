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


int main(){
    int cols, rows, channels;    
    Matrix<double, Dynamic, Dynamic, RowMajor> A = loadImage(&cols, &rows, &channels);

    //Here starts task 1
    cout << "---------TASK1---------" << endl;

    MatrixXd traspose = A.transpose();
    MatrixXd ATA = traspose*A;
    cout << "The norm of A'A is: " << ATA.norm() << endl;

    //Here starts task 2
    cout << "---------TASK2---------" << endl;
    
    int lambda1 = computeSingularValues(A);
    cout << "lambda1: " << lambda1 << endl;
    
    
    //Here starts task 3
    cout << "---------TASK3---------" << endl;
    
    Eigen::saveMarket(ATA,"matrix_ATA.mtx");

    system("mpirun -n 4 ./lis-2.1.6/test/etest1 matrix_ATA.mtx -e 1 -etol 1.0e-8 > maxEigenLisOutput.txt");

    string maxEigenValue = readFile("maxEigenLisOutput.txt", "eigenvalue");
    cout << maxEigenValue << endl;


    //Here starts task 4
    cout << "---------TASK4---------" << endl;
    for(int shifting=0;shifting<16;shifting+=2)
    
    
    // string command = ("mpirun -n 4 ./lis-2.1.6/test/etest1 matrix_ATA.mtx -e 1 -etol 1.0e-8 -shift ").append(to_string(shifting)).append(" > maxEigenLisOutput.txt");
    // system(command);
    // string maxEigenValue = readFile("maxEigenLisOutput.txt", "eigenvalue");
    // cout << maxEigenValue << endl;


    //Here starts task 5
    cout << "---------TASK5---------" << endl;

    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    Eigen::VectorXd singularValues = svd.singularValues();
    int sizeS = singularValues.size();
    MatrixXd S = MatrixXd::Zero(sizeS, sizeS);
    for (int i = 0; i < sizeS; ++i) {
        S(i, i) = singularValues(i);
    }
    cout << "Euclidian norm of S as vector:" << singularValues.norm() << endl;
    cout << "Euclidian norm of S as matrix:" << S.norm() << endl;

    cout << "ANSWER: Real euclidian norm of S:" << singularValues(0) << endl;

    
    //Here starts task 6
    cout << "---------TASK6---------" << endl;

    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::MatrixXd C1 = U.leftCols(40);
    Eigen::MatrixXd SV = S * V.transpose();
    Eigen::MatrixXd D1 = SV.leftCols(40);

    Eigen::MatrixXd C2 = U.leftCols(80);
    Eigen::MatrixXd D2 = SV.leftCols(80);

    cout << "Non-zero C1:" << C1.nonZeros() << endl;
    cout << "Non-zero D1:" << D1.nonZeros() << endl;
    cout << "Non-zero C2:" << C2.nonZeros() << endl;
    cout << "Non-zero D2:" << D2.nonZeros() << endl;


    //Here starts task 7
    cout << "---------TASK7---------" << endl;




    //Here starts task 8
    cout << "---------TASK8---------" << endl;




    //Here starts task 9
    cout << "---------TASK9---------" << endl;





    //Here starts task 10
    cout << "---------TASK10---------" << endl;




    //Here starts task 11
    cout << "---------TASK11---------" << endl;




    //Here starts task 12
    cout << "---------TASK12---------" << endl;




    //Here starts task 13
    cout << "---------TASK13---------" << endl;
}

