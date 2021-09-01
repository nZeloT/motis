#pragma once

#include <cstdio>
#include <string>
#include <vector>

extern "C" {

struct cudaDeviceProp;

void print_device_properties(cudaDeviceProp const&);
int set_device(std::vector<std::string> const&);
}