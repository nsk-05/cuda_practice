#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

int main(){

    std:srand(static_cast<unsigned int>(std::time(0)));

    std::ofstream outFile("../random_numbers.txt");

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the file!" << std::endl;
        return 1;
    }

    int i=0;
    while (i<1000)
    {
        int randomNum = (std::rand() % 1000) + 1; // Generate random number
        outFile << randomNum << std::endl; // Write the random number to the file
        i++;
    }
    
    outFile.close();
    return 0;
}

