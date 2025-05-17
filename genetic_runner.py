import random, time, json
import numpy as np
from environment import UAVVictimSearchEnv
import json
import numpy as np

ACTION_SPACE = [0, 1, 2, 3]  # up, down, left, right

def create_chromosome(length):
    return [(random.choice(ACTION_SPACE), random.choice(ACTION_SPACE)) for _ in range(length)]

# def evaluate(env, chromosome):
#     obs = env.reset()
#     total_reward = 0
#     for actions in chromosome:
#         _, reward, done, _ = env.step(actions)
#         total_reward += reward
#         if done: break
#     return total_reward, np.sum(env.visited), env.visited.copy()

# def evaluate(env, chromosome):
#     obs = env.reset()
#     total_reward = 0
#     visit_counts = np.zeros_like(env.visited, dtype=np.int32)

#     for actions in chromosome:
#         _, reward, done, _ = env.step(actions)
#         # Increment visits for both UAVs
#         visit_counts[tuple(env.uav1_pos)] += 1
#         visit_counts[tuple(env.uav2_pos)] += 1

#         total_reward += reward
#         if done:
#             break

#     return total_reward, np.sum(visit_counts > 0), visit_counts

def evaluate(env, chromosome):
    obs = env.reset()
    total_reward = 0
    visit_counts = np.zeros_like(env.visited, dtype=np.int32)
    out_of_bounds_moves = 0

    for actions in chromosome:
        # Store previous positions
        prev_uav1 = tuple(env.uav1_pos)
        prev_uav2 = tuple(env.uav2_pos)

        # Simulate move to see if it goes out of bounds
        for i, (uav_pos, action) in enumerate(zip([env.uav1_pos, env.uav2_pos], actions)):
            before = uav_pos[:]
            env._move_uav(uav_pos, action)
            if uav_pos == before:
                out_of_bounds_moves += 1  # Move had no effect, thus hit boundary

        visit_counts[tuple(env.uav1_pos)] += 1
        visit_counts[tuple(env.uav2_pos)] += 1

        _, reward, done, _ = env.step(actions)
        total_reward += reward
        if done:
            break

    unique_cells = np.sum(visit_counts > 0)
    overlaps = np.sum(visit_counts > 1)

    # Apply new fitness formula
    fitness = (
        5 * unique_cells
        + total_reward  # from victim count
        - 3 * out_of_bounds_moves
        - 2 * overlaps
    )

    return fitness, unique_cells, visit_counts, total_reward


def genetic_search(config):
    env = UAVVictimSearchEnv()
    population = [create_chromosome(config["chromosome_length"]) for _ in range(config["population"])]
    best_result = {"fitness": 0, "chromosome": None, "visited": None, "victim_count":0}
    for gen in range(config["generations"]):
        fitnesses = []
        visited_snapshots = []
        victim_Counts_lst = []
        for chrom in population:
            reward, visited_count, visited, victim_count = evaluate(env, chrom)
            fitnesses.append(reward + visited_count)
            visited_snapshots.append(visited)
            victim_Counts_lst.append(victim_count)
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_result["fitness"]:
            best_result.update({
                "fitness": fitnesses[best_idx],
                "chromosome": population[best_idx],
                "visited": visited_snapshots[best_idx],
                "victim_count": victim_Counts_lst[best_idx]
            })

        elites = sorted(zip(fitnesses, population), reverse=True)[:config["elitism_count"]]
        new_population = [chrom for _, chrom in elites]

        while len(new_population) < config["population"]:
            p1, p2 = random.sample(elites, 2)
            cut = random.randint(1, config["chromosome_length"] - 1)
            child1 = p1[1][:cut] + p2[1][cut:]
            child2 = p2[1][:cut] + p1[1][cut:]
            new_population.extend([mutate(child1, config["mutation_rate"]), mutate(child2, config["mutation_rate"])])

        population = new_population[:config["population"]]
    return best_result, env.victim_positions

def mutate(chromosome, mutation_rate):
    return [(random.choice(ACTION_SPACE), random.choice(ACTION_SPACE)) if random.random() < mutation_rate else gene for gene in chromosome]

def run_all_configs(config_path=r"C:\Users\vdgar\OneDrive\Desktop\Gym_Environment\Experiment\configurations.json", render_best=True):
    with open(config_path) as f:
        configs = json.load(f)

    results = []
    overall_best = {"fitness": -float("inf"), "chromosome": None, "config": None}

    for config in configs:
        print(f"\nRunning {config['name']}...")
        start_time = time.time()
        result, victim_positions = genetic_search(config)
        elapsed = time.time() - start_time

        # results.append({
        #     "name": config["name"],
        #     "victims_found": result["fitness"],
        #     "cells_covered": np.sum(result["visited"]),
        #     "time_taken": float(elapsed),
        #     "visited": result["visited"],
        #     "victim_positions": victim_positions,
        #     "chromosome": result["chromosome"]
        # })
        results.append({
            "name": config["name"],
            "victims_found": result["victim_count"],
            "cells_covered": np.sum(result["visited"] > 0),
            "time_taken": elapsed,
            "visited": result["visited"],  # Heatmap of frequencies now!
            "victim_positions": victim_positions,
            "chromosome": result["chromosome"]
        })


        if result["fitness"] > overall_best["fitness"]:
            overall_best.update({
                "fitness": result["fitness"],
                "chromosome": result["chromosome"],
                "config": config
            })

    # ✅ Render best policy from all configs
    if render_best and overall_best["chromosome"]:
        with open(r"C:\Users\vdgar\OneDrive\Desktop\Gym_Environment\Experiment\Results\best_policy.json", "w") as f:
            json.dump({
                "config": overall_best["config"],
                "fitness": int(overall_best["fitness"]),  # 👈 cast to Python int
                "chromosome": overall_best["chromosome"]
            }, f, indent=4)


        # Save visited map (heatmap) as .npy
        np.save(r"C:\Users\vdgar\OneDrive\Desktop\Gym_Environment\Experiment\Results\best_policy_visited.npy", result["visited"])

        print("✅ Best policy saved to 'best_policy.json' and 'best_policy_visited.npy'")
        print(f"\nRendering BEST policy from '{overall_best['config']['name']}' with fitness {overall_best['fitness']}")
        env = UAVVictimSearchEnv()
        env.reset()
        for actions in overall_best["chromosome"]:
            env.step(actions)
            env.render()
        env.close()

    return results



if __name__ == "__main__":
    final_results = run_all_configs()
