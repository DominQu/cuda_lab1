#include <iostream>
#include <cstdint>
#include <math.h>
#include <chrono>
#include <climits>


// CPU section

// troche mniej naiwna metoda
// bazuje na założeniu że wszystkie liczby pierwsze są postaci (6n +- 1)
bool CPUprime(uint64_t n){
    auto start = std::chrono::high_resolution_clock::now();

    if (n <= 3){
        return (n > 1);
    }
    if (n % 2 == 0 || n % 2 == 0){
        return false;
    }
    // uint64_t i = 5;
    for(int i = 5; i < std::ceil(std::sqrt(n)); i+=6) {
        if(n % i == 0 || n % (i +2) == 0)
        {
            auto stop = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "current number: " << duration.count() << " milliseconds" << std::endl;
            return false; 
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "current number: " << duration.count() << " milliseconds" << std::endl;

    return true;
}


void testCPU(bool (*func)(uint64_t)){

    uint64_t num[6] = {524287, 2147483647, 2305843009213693951, 274876858369, 4611686014132420609, 1125897758834689 };
    bool res[6];

    std::cout << "CPU primality test:\n";
    auto start = std::chrono::high_resolution_clock::now();    

    for(int i = 0; i < 6; i++){
        res[i] = func(num[i]);
        std::cout << "Is number " << num[i] << " prime?: ";
        if(res[i] == 1){
            std::cout << " Yes it is" << std::endl;
        }
        else{
            std::cout << " No it isn't" << std::endl;

        }
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CPU test duration: " << duration.count() << " milliseconds" << std::endl;

}
// GPU section

void testGPU(bool (*func)(uint64_t)){

    const int testlen = 3;
    uint64_t num[testlen] = {524287, 2305843009213693951, 4611686014132420609 };
    bool res[testlen];

    std::cout << "GPU primality test:\n";
    auto start = std::chrono::high_resolution_clock::now();    

    for(int i = 0; i < testlen; i++){
        res[i] = func(num[i]);
        std::cout << "Is number " << num[i] << " prime?: ";
        if(res[i] == 1){
            std::cout << " Yes it is" << std::endl;
        }
        else{
            std::cout << " No it isn't" << std::endl;

        }
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "GPU test duration: " << duration.count() << " milliseconds" << std::endl;

}

__global__ void ker_GPUprime(uint64_t* num, uint32_t* res, uint32_t* maxind){

    for(uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        5 + i * 6 <= *maxind - 2;
        i += blockDim.x + gridDim.x )
        {
            uint32_t realindex = 5 + i * 6;

            if(*num % realindex == 0 || *num % (realindex+2) == 0)
            {
                uint32_t block = i / 32;
                res[block] = 0;
            }
        }
}


bool GPUprime(uint64_t num){

    uint32_t sqrtnum = (uint32_t)std::floor(std::sqrt(num));
    uint32_t bitnum = ((sqrtnum - 5) / 6 + 1);   //number of threads
    uint32_t reslen = (bitnum / 64 + (bitnum % 64 != 0)) * 2;
    uint32_t *res = new uint32_t[reslen];

    for(int i = 0 ; i < reslen; i++){
        res[i] = (UINT_MAX);
    }

    uint64_t* dnum;
    uint32_t* dres;
    uint32_t* maxind;

    auto start = std::chrono::high_resolution_clock::now();
    cudaMalloc(&maxind, 32);
    cudaMalloc(&dnum, 64);
    cudaMalloc(&dres, reslen*4);
    cudaMemcpy(maxind, &sqrtnum, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dnum, &num, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(dres, res, reslen*4, cudaMemcpyHostToDevice);

    uint32_t threads = 512;
    dim3 gridsize = {bitnum/threads + (bitnum%threads !=0)};

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    // std::cout << "number od sms: "<< numSMs << std::endl;
    // std::cout << "bitnum: " << bitnum << std::endl;

    
    ker_GPUprime<<<numSMs, threads>>>(dnum, dres, maxind);

    cudaMemcpy(res, dres, reslen*4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    bool prime = true;
    for(int j = 0; j < reslen ;j++){
        
        if(res[j] != UINT_MAX){
            prime = false;
        }
    }

    cudaFree(dnum);
    cudaFree(dres);
    cudaFree(maxind);
    delete[] res;


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "current number: " << duration.count() << " milliseconds" << std::endl;

    return prime;

}



int main() {

    // CPU test
    // testCPU(&CPUprime);

    // Gpu test
    testGPU(&GPUprime);

}