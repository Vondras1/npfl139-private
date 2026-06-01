// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <map>
#include <array>
#include <functional>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>


#include "board_game.h"

template<BoardGame G>
using Policy = std::array<float, G::ACTIONS>;

template<BoardGame G>
using Evaluator = std::function<void(const G&, Policy<G>&, float&)>;

//########
//# MCTS #
//########
template<BoardGame G>
struct MCTNode {
  float prior;
  G game;
  bool has_game;
  std::map<int, MCTNode<G>> children;
  int visit_count;
  double total_value;

  MCTNode(float prior = 0.0f) : prior(prior), has_game(false), visit_count(0), total_value(0.0) {}
};

template<BoardGame G>
double value(const MCTNode<G>& node){
  if (node.visit_count == 0){
    return 0;
  }
  else{
    return (node.total_value/node.visit_count);
  }
}

template<BoardGame G>
bool is_evaluated(const MCTNode<G>& node){
  return node.visit_count > 0;
}

template<BoardGame G>
double outcome_value(Outcome outcome) {
  // WIN = 3, DRAW = 2, LOSS = 1
  // Convert to: WIN -> 1, DRAW -> 0, LOSS -> -1
  return int(outcome) - 2;
}

template<BoardGame G>
void evaluate_node(
    MCTNode<G>& node,
    const G& game,
    const Evaluator<G>& evaluator
) {
  node.game = game;
  node.has_game = true;

  double node_value;

  Outcome outcome = game.outcome(game.to_play);

  if (outcome != Outcome::UNFINISHED) {
    node_value = outcome_value<G>(outcome);
  } else {
    Policy<G> policy;
    float predicted_value;

    evaluator(game, policy, predicted_value);
    node_value = predicted_value;

    // Normalize priors only over valid actions.
    float sum_priors = 0.0f;
    for (int action = 0; action < G::ACTIONS; ++action) {
      if (game.valid(action)) {
        sum_priors += policy[action];
      }
    }

    for (int action = 0; action < G::ACTIONS; ++action) {
      if (game.valid(action)) {
        float prior;
        if (sum_priors > 0.0f)
          prior = policy[action] / sum_priors;
        else
          prior = 1.0f; // temporarily, fixed below if needed

        node.children.emplace(action, MCTNode<G>(prior));
      }
    }

    if (sum_priors <= 0.0f && !node.children.empty()) {
      float uniform = 1.0f / node.children.size();
      for (auto& [action, child] : node.children) {
        child.prior = uniform;
      }
    }
  }

  node.total_value += node_value;
  node.visit_count += 1;
}

template<BoardGame G>
void add_exploration_noise(MCTNode<G>& node, float epsilon, float alpha) {
  if (node.children.empty())
    return;

  std::gamma_distribution<float> gamma(alpha, 1.0f);

  std::vector<float> noise;
  noise.reserve(node.children.size());

  float noise_sum = 0.0f;

  for (size_t i = 0; i < node.children.size(); ++i) {
    float sample = gamma(*board_game_generator);
    noise.push_back(sample);
    noise_sum += sample;
  }

  size_t i = 0;
  for (auto& [action, child] : node.children) {
    float noise_value = noise[i] / noise_sum;
    child.prior = epsilon * noise_value + (1.0f - epsilon) * child.prior;
    ++i;
  }
}

template<BoardGame G>
std::pair<int, MCTNode<G>*> select_child(MCTNode<G>& node) {
  int best_action = -1;
  MCTNode<G>* best_child = nullptr;
  double best_score = -std::numeric_limits<double>::infinity();

  for (auto& [action, child] : node.children) {
    double Q = -value(child);

    double C = std::log((1.0 + node.visit_count + 1965.2) / 1965.2) + 1.25;

    double H = std::sqrt(node.visit_count) / (child.visit_count + 1.0);

    double score = Q + C * child.prior * H;

    if (score > best_score) {
      best_score = score;
      best_action = action;
      best_child = &child;
    }
  }

  return {best_action, best_child};
}

template<BoardGame G>
void mcts(
    const G& game,
    const Evaluator<G>& evaluator,
    int num_simulations,
    float epsilon,
    float alpha,
    Policy<G>& policy
) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.

  MCTNode<G> root;
  evaluate_node(root, game, evaluator);

  add_exploration_noise(root, epsilon, alpha);

  for (int simulation = 0; simulation < num_simulations; ++simulation) {
    MCTNode<G>* node = &root;

    std::vector<MCTNode<G>*> search_path;
    std::vector<int> action_path;

    search_path.push_back(node);

    while (!(node->children.empty())) {
      auto [action, child] = select_child(*node);

      node = child;
      search_path.push_back(node);
      action_path.push_back(action);
    }

    double node_value;

    if (!is_evaluated(*node)) {
      G node_game;

      if (search_path.size() == 1) {
        node_game = game;
      } else {
        MCTNode<G>* parent = search_path[search_path.size() - 2];
        int action = action_path.back();

        node_game = parent->game;
        node_game.move(action);
      }

      evaluate_node(*node, node_game, evaluator);
      node_value = value(*node);
    } else {
      Outcome outcome = node->game.outcome(node->game.to_play);
      node_value = outcome_value<G>(outcome);

      node->total_value += node_value;
      node->visit_count += 1;
    }

    for (int i = int(search_path.size()) - 2; i >= 0; --i) {
      node_value = -node_value;
      search_path[i]->visit_count += 1;
      search_path[i]->total_value += node_value;
    }
  }

  policy.fill(0.0f);

  float visit_sum = 0.0f;

  for (const auto& [action, child] : root.children) {
    policy[action] = float(child.visit_count);
    visit_sum += policy[action];
  }

  if (visit_sum > 0.0f) {
    for (float& p : policy) {
      p /= visit_sum;
    }
  }
}

