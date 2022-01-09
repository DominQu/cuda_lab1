#include <iostream>
#include <cstdint>
#include <math.h>
#include <chrono>
#include <climits>
#include <omp.h>

// CPU section

bool CPUprime(uint64_t n){
    auto start = std::chrono::high_resolution_clock::now();

    // check 2 and 3
    if (n <= 3){
        return (n > 1);
    }
    // check if number isn't even
    if (n % 2 == 0){
        return false;
    }

    // loop through every six number less than sqrt(num) starting at 5
    for(int i = 5; i < std::ceil(std::sqrt(n)); i+=6) {
        if(n % i == 0 || n % (i +2) == 0)
        {
            auto stop = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "Test duration: " << duration.count() << " microseconds ";
            return false; 
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Test duration: " << duration.count() << " microseconds ";

    return true;
}

void concurrentTest(const uint64_t& n, uint32_t& res, int threadId, int numthreads, const uint32_t& maxind){
    
    for(uint32_t i = threadId;
        5 + i * 6 <= maxind - 2;
        i += numthreads)
        {
            uint32_t realindex = 5 + i * 6;

            if(n % realindex == 0 || n % (realindex+2) == 0)
            {
                res = 0;
            }
        }
}
bool concurrentPrime(uint64_t n){

    auto start = std::chrono::high_resolution_clock::now();

    uint32_t res = 1;
    uint32_t maxind = (uint32_t)std::floor(std::sqrt(n));
    int threadId;
    int numthreads;
#pragma omp parallel default(shared) private(threadId, numthreads)
    {
        threadId = omp_get_thread_num();
        numthreads = omp_get_num_threads();

        concurrentTest(n, res, threadId, numthreads, maxind);
    }
    if( res == 0){
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Test duration: " << duration.count() << " microseconds ";
        return false;
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Test duration: " << duration.count() << " microseconds ";

    return true;
}


void testCPU(bool (*func)(uint64_t)){
    // test numbers given in the task

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

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "CPU test duration: " << duration.count() << " microseconds" << std::endl;

}

// GPU section


__global__
void dev_GPUgridstride(uint64_t* num, uint32_t* res, uint32_t* maxind){
    // use grid-stride loop to reuse threads

    for(uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        5 + i * 6 <= *maxind - 2;
        i += blockDim.x * gridDim.x )
        {
            uint32_t realindex = 5 + i * 6;

            if(*num % realindex == 0 || *num % (realindex+2) == 0)
            {
                *res = 0;
            }
        }
}

__global__ void dev_GPUmonolithic(uint64_t* num, uint32_t* res, uint32_t* maxind){

    uint32_t index = (threadIdx.x + blockIdx.x * blockDim.x);
    uint32_t realindex = 5 + index * 6;

    if(realindex <= *maxind-2){
        if(*num % realindex == 0 || *num % (realindex+2) == 0)
        {
            *res = 0;
        }
    }
}

bool GPUprime(uint64_t num, bool gridstride){

    auto start = std::chrono::high_resolution_clock::now();

    uint32_t sqrtnum = (uint32_t)std::floor(std::sqrt(num));
    uint32_t *res = new uint32_t;
    *res = 1;

    uint64_t* dnum;
    uint32_t* dres;
    uint32_t* maxind;


    // check 2 and 3
    if (num <= 3){
        return (num > 1);
    }
    // check if number isn't even
    if (num % 2 == 0){
        return false;
    }

    // allocate CUDA memory
    cudaMalloc(&maxind, 4);
    cudaMalloc(&dnum, 8);
    cudaMalloc(&dres, 4);
    cudaMemcpy(maxind, &sqrtnum, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dnum, &num, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dres, res, 4, cudaMemcpyHostToDevice);

    if(gridstride == 1){

        // number of threads and blocks
        uint32_t threads = 1024;
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

        // kernel call
        dev_GPUgridstride<<<4*numSMs, threads>>>(dnum, dres, maxind);
        cudaDeviceSynchronize();
    }
    else{
        uint32_t bitnum = ((sqrtnum - 5) / 6 + 1);

        dim3 blocksize= {32};
        dim3 gridsize = {bitnum/32 + (bitnum%32 !=0)};
        
        dev_GPUmonolithic<<<gridsize, blocksize>>>(dnum, dres, maxind);
        cudaDeviceSynchronize();
    }

    // copy the solution and check it
    cudaMemcpy(res, dres, 4, cudaMemcpyDeviceToHost);

    bool prime = true;
    if(*res == 0){
        prime = false;
    }

    // deallocate the memory
    cudaFree(dnum);
    cudaFree(dres);
    cudaFree(maxind);
    delete res;

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Test duration: " << duration.count() << " microseconds ";

    return prime;

}


void testGPU(bool (*func)(uint64_t, bool), bool gridstride = 1){
    // test numbers given in the task

    const int testlen = 6;
    uint64_t num[testlen] = {524287, 2147483647, 2305843009213693951, 274876858369, 4611686014132420609, 1125897758834689 };
    bool res[testlen];
    cudaDeviceSynchronize();


    std::cout << "GPU primality test: ";
    if(gridstride == 1){
        std::cout << "grid-stride version" << std::endl;
    }
    else{
        std::cout << "monolithic kernel version" << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();    

    for(int i = 0; i < testlen; i++){
        res[i] = func(num[i], gridstride);
        std::cout << "Is number " << num[i] << " prime?: ";
        if(res[i] == 1){
            std::cout << " Yes it is" << std::endl;
        }
        else{
            std::cout << " No it isn't" << std::endl;

        }
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "GPU test duration: " << duration.count() << " microseconds" << std::endl;

}

void usertest(){

    uint64_t num;
    std::cout << "Podaj liczbe do testu: " << std::endl;
    std::cin >> num;
    bool cpures;
    bool gpures;

    std::cout << "\nCPU standard" << std::endl;
    cpures = CPUprime(num);
    if(cpures ==1){
        std::cout << "(CPU) Number " << num << " is prime" << std::endl;
    }
    else{
        std::cout << "(CPU) Number " << num << " is not prime"<< std::endl;
    }

    std::cout << "CPU concurrent" << std::endl;
    cpures = concurrentPrime(num);
    if(cpures ==1){
        std::cout << "(CPU) Number " << num << " is prime" << std::endl;
    }
    else{
        std::cout << "(CPU) Number " << num << " is not prime"<< std::endl;
    }

    cudaDeviceSynchronize();
    std::cout << "GPU grid-stride loop" << std::endl;
    gpures = GPUprime(num, 1);
    if(gpures ==1){
        std::cout << "(GPU) Number " << num << " is prime"<< std::endl;
    }
    else{
        std::cout << "(GPU) Number " << num << " is not prime"<< std::endl;
    }

    std::cout << "GPU monolithic kernel" << std::endl;
    gpures = GPUprime(num, 0);
    if(gpures ==1){
        std::cout << "(GPU) Number " << num << " is prime"<< std::endl;
    }
    else{
        std::cout << "(GPU) Number " << num << " is not prime"<< std::endl;
    }
}


int main() {

    // Aby przetestować liczby podane w instrukcji należy odkomentować poniższe funkcje.
    // Funkcja testGPU może przyjąć jeden dodatkowy argument typu bool, który pozwala wybrać 
    // czy używane będą monolityczne kernele czy pętla grid-stride.
    bool test = false;
    std::cout << "Wybierz rodzaj testu (wpisz 0 lub 1):\n0 - test automatyczny\n1 - testuj liczbe" << std::endl;
    std::cin >> test;
    if(test == 0) {

        std::cout << "CPU concurrent" << std::endl;
        testCPU(&concurrentPrime);
        std::cout << "CPU standard" << std::endl;
        testCPU(&CPUprime);
        testGPU(&GPUprime);
        testGPU(&GPUprime, false);
    }
    else {
        usertest();
    }


    


}
