# Optimizing Leaf Raking

Now that it’s fall, it’s leaf‑raking season. And this year, as I raked my yard, I realized it was basically one big optimization problem.

When we first moved in, our yard was full of sweet gum balls, so I raked them all from the back of the yard to the front and bagged them on the sidewalk. That made sense because it was easier to bag from there. But leaves are different — they don’t “roll” downhill well, and our city doesn’t pick them up from the street. Instead, when raking my leaves, I made four piles: one on either side of the tree, one near the front, and one on the far side of the yard to catch the stray leaves.

As I worked, I started wondering how far from the optimal solution this was. Would fewer piles be faster because I’d bag everything at once? Or more piles, so I wouldn’t have to rake as far? Maybe a single back‑to‑front pass would minimize the total time it took to rake the leaves?

At that point, my brain stopped raking and started modeling.

## Problem Definition

To minimize total raking time, I wanted to answer two questions:

- How many piles should I form?
- Where should they go?

Because both the number of piles and their locations are discrete decisions, while time and pile size are continuous, I realized this would need to be modeled as a mixed‑integer linear program (MILP) — a type of optimization problem that combines binary (“yes/no”) and continuous variables.

## Parameters and Assumptions

At first, I thought this would be simple — I’d just need my raking speed and bagging time. But that quickly grew into a longer list of parameters:

- Raking speed: depends on both leaf density and pile size. Sparse areas are fast; dense ones slow you down.
- Bagging time: not constant — larger piles take disproportionately less time to bag.
- Leaf distribution: varies across the yard (dense under the tree, thinning toward the edges).
- Yard grid: divided into 18 × 18‑inch squares (the width of my rake).
- Wind direction: mostly from the north, which shifts leaf accumulation southward.
- Slope: ignored for simplicity, though it would affect real‑world performance.

Placeholder: Table 1 – Parameters and their estimated values/measurement plans.

## Experimental Design (Conceptually)

If I were to truly measure all this, I’d conduct a small experiment. I’d evenly spread leaves over a 100‑ft section, mark every 10 ft, and record myself raking. Later, I could measure how raking speed changed with leaf density and distance.

Similarly, I could time bagging for piles of different sizes (¼, ½, ¾, and full) to fit a function estimating bagging time as a function of pile mass.

Of course, I didn’t actually rake and time four entire yard‑loads or weigh the leaves — I’m a nerd, but not quite that much of a nerd (yet). Instead, I made rough estimates for each parameter and used those to build the model. The good news: even if the numbers are off, the model structure is still valid. If someone later measured these parameters precisely, we’d just plug in the new values — no need to rebuild the model.

## Problem Formulation

Objective: Minimize total time spent raking and bagging leaves.

### Decision Variables

- [Define binary variables for pile placement]
- [Define continuous variables for pile mass and raking time]

Constraints (conceptual):

- Each bag can only hold a maximum mass.
- All piles must be bagged.
- Total leaf mass in all piles = total mass of leaves in yard.

Placeholder: Equations (1)–(4) — Full mathematical model.

## Alternative Strategies for Comparison

To test the optimization’s performance, I compared it against three simple heuristic strategies:

- Back‑to‑Front: rake continuously from back to front, depositing piles as you go.
- Micro‑Piles: make many small piles near dense leaf areas to minimize raking distance.
- Centered‑Piles: create a few large piles in central locations and rake everything toward them.

The optimization model should, in theory, outperform all of these — but comparing them helps illustrate why optimization matters.

## Results and Analysis

For a yard measuring 40 ft × 60 ft, the model found the optimal plan involved [x] piles in [y] locations, balancing raking distance against bagging time.

| Strategy        | Total Time (min) | % Faster vs. Worst | Pile Count |
|-----------------|------------------|--------------------|------------|
| Optimization    | …                | …                  | …          |
| Micro‑Piles     | …                | …                  | …          |
| Back‑to‑Front   | …                | …                  | …          |
| Centered‑Piles  | …                | …                  | …          |

Placeholder: Table 3 – Comparison of raking strategies.

Placeholder: Figure 1 – Heatmap of optimal pile locations.

Interestingly, the optimized solution wasn’t dramatically faster than the micro‑pile method — just more balanced. Micro‑piles performed well because most of the work comes from local leaf movement rather than long‑distance raking.

## Conclusion

The optimization model produced the fastest raking plan, but not by much — the micro‑piles strategy came close, making it a surprisingly efficient “human heuristic.”

Of course, nobody is going to run a MILP solver before heading outside with a rake. The real value of this exercise isn’t to optimize leaf‑raking — it’s to show how analytical methods can uncover trade‑offs hidden in everyday tasks. Even something as simple as cleaning up a yard involves location decisions, distance trade‑offs, and workload balancing — the same principles behind facility‑location models, delivery logistics, and warehouse layout problems.
