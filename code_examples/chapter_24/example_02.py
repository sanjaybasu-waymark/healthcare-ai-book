"""
Chapter 24 - Example 2
Extracted from Healthcare AI Implementation Guide
"""

import simpy
import random

class HealthcareSystem:
    def __init__(self, env, num_doctors, service_time_mean):
        self.env = env
        self.doctors = simpy.Resource(env, capacity=num_doctors)
        self.service_time_mean = service_time_mean
        self.wait_times = [] \# To collect wait times for analysis

    def patient_arrival(self, name):
        arrival_time = self.env.now
        \# print(f"Patient {name} arrived at {arrival_time:.2f}") \# Commented for cleaner output

        with self.doctors.request() as request:
            yield request
            wait_time = self.env.now - arrival_time
            self.wait_times.append(wait_time)
            \# print(f"Patient {name} started consultation at {self.env.now:.2f} after waiting {wait_time:.2f}") \# Commented for cleaner output
            yield self.env.timeout(random.expovariate(1.0 / self.service_time_mean))
            \# print(f"Patient {name} finished consultation at {self.env.now:.2f}") \# Commented for cleaner output

def setup(env, num_doctors, service_time_mean, arrival_interval_mean, num_patients):
    healthcare_system = HealthcareSystem(env, num_doctors, service_time_mean)

    for i in range(num_patients):
        env.process(healthcare_system.patient_arrival(f"Patient_{i}"))
        yield env.timeout(random.expovariate(1.0 / arrival_interval_mean))
    return healthcare_system \# Return the system to access collected metrics

if __name__ == "__main__":
    print("Running basic queuing simulation for a healthcare system...")

    \# Simulation parameters
    RANDOM_SEED = 42
    NUM_DOCTORS = 2
    SERVICE_TIME_MEAN = 10  \# minutes per patient
    ARRIVAL_INTERVAL_MEAN = 7  \# minutes between patient arrivals
    NUM_PATIENTS = 100 \# Increased number of patients for better statistics

    random.seed(RANDOM_SEED)

    \# Create a SimPy environment
    env = simpy.Environment()

    \# Start the setup process and get the healthcare system instance
    system_instance = env.process(setup(env, NUM_DOCTORS, SERVICE_TIME_MEAN, ARRIVAL_INTERVAL_MEAN, NUM_PATIENTS))

    \# Run the simulation
    env.run(until=system_instance) \# Run until all patients have arrived and been processed

    print("\nSimulation finished.")

    \# Analyze collected metrics
    if system_instance.is_alive:
        \# If the setup process is still alive, it means not all patients have been processed
        \# This can happen if the simulation time is too short or if the system is overloaded.
        print("Warning: Simulation ended before all patients were processed. Consider increasing simulation time or reducing patient count.")
    
    if system_instance.value and system_instance.value.wait_times:
        avg_wait_time = sum(system_instance.value.wait_times) / len(system_instance.value.wait_times)
        print(f"Average patient wait time: {avg_wait_time:.2f} minutes")
        print(f"Maximum patient wait time: {max(system_instance.value.wait_times):.2f} minutes")
    else:
        print("No wait times recorded, possibly due to no patients or immediate service.")

    \# Error handling example: Invalid parameters
    print("\n--- Example with invalid simulation parameters ---")
    try:
        \# Attempt to create a system with zero doctors
        env_err = simpy.Environment()
        setup(env_err, 0, SERVICE_TIME_MEAN, ARRIVAL_INTERVAL_MEAN, NUM_PATIENTS)
    except ValueError as e:
        print(f"Caught expected error for invalid parameters: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")