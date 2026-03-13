#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // Round i (1-based): process keys[0..i] and values[0..i] with current_query
    // current_query is shape [i+1, d]
    // keys[j] and values[j] are shape [1, d]

    // Concatenate all keys vertically to get K of shape [i+1, d]
    Matrix* K = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (K == nullptr) {
        K = matrix_memory_allocator.Allocate("K_concat");
        gpu_sim.Copy(keys[j], K, kInGpuHbm);
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("K_temp");
        gpu_sim.Concat(K, keys[j], temp, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(K);
        K = temp;
      }
    }

    // Concatenate all values vertically to get V of shape [i+1, d]
    Matrix* V = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (V == nullptr) {
        V = matrix_memory_allocator.Allocate("V_concat");
        gpu_sim.Copy(values[j], V, kInGpuHbm);
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("V_temp");
        gpu_sim.Concat(V, values[j], temp, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(V);
        V = temp;
      }
    }

    // Move Q, K, V to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K);
    gpu_sim.MoveMatrixToSharedMem(V);

    // Transpose K to get K^T
    gpu_sim.Transpose(K, kInSharedMemory);

    // Compute Q * K^T = [i+1, d] * [d, i+1] = [i+1, i+1]
    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K, QK);

    // Release K as we don't need it anymore
    gpu_sim.ReleaseMatrix(K);

    // Compute Softmax row-wise on QK
    // Softmax(x)[j] = exp(x[j]) / sum(exp(x[j]))
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);
    gpu_sim.ReleaseMatrix(QK);

    // For each row, compute sum of exponentials
    Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax");
    for (size_t row = 0; row <= i; ++row) {
      Matrix* row_data = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(QK_exp, row, row_data, kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_data, row_sum);

      Matrix* normalized_row = matrix_memory_allocator.Allocate("norm_row");
      gpu_sim.MatDiv(row_data, row_sum, normalized_row);

      if (row == 0) {
        gpu_sim.Copy(normalized_row, softmax_result, kInSharedMemory);
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("softmax_temp");
        gpu_sim.Concat(softmax_result, normalized_row, temp, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        softmax_result = temp;
      }

      gpu_sim.ReleaseMatrix(row_data);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(normalized_row);
    }

    gpu_sim.ReleaseMatrix(QK_exp);

    // Compute Softmax(QK) * V = [i+1, i+1] * [i+1, d] = [i+1, d]
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_result, V, result);

    gpu_sim.ReleaseMatrix(softmax_result);
    gpu_sim.ReleaseMatrix(V);

    // Move result back to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu