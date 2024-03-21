import multiprocessing

class Parallel:
    @staticmethod
    def evaluate_in_parallel(Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
        # print('Xs_root_item in eval config : ',Xs_root_item)
        agents = []
        return self.ei_roll.sample(sample, Xs_root_item, agents, num_agents, self.tf, globalXtrain, self.horizon, globalYtrain, region_support, model, rng)


    @classmethod
    def parallel_method(cls, args_list):
        # Create a multiprocessing Pool with the number of desired processes
        pool = multiprocessing.Pool()

        # Map the worker function to each argument in parallel
        results = pool.map(cls._worker_function, args_list)

        # Close the pool to free up resources
        pool.close()
        pool.join()

        return results

# if __name__ == "__main__":
#     my_instance = MyClass()

#     # Define a list of arguments to be processed in parallel
#     args_list = [1, 2, 3, 4, 5]

#     # Call the parallel_method to execute the worker function in parallel
#     results = my_instance.parallel_method(args_list)

#     print("Results:", results)
