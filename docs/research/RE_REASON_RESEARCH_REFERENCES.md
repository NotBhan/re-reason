# RE:Reason Research References

Curated literature for experiments on rule generalisation, modular arithmetic, recurrent state, variable binding, and algorithmic reasoning.

## How to use this bibliography

These papers motivate hypotheses and methods; they do not prove a particular RE:Reason mechanism is correct. In reports, distinguish prior results, project observations, and untested hypotheses. Results from MLPs, transformers, or external-memory architectures are not direct evidence about RE:Reason's MGU-style recurrent core.

## Priority 1 — Generalisation and modular arithmetic

### 1. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
**Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra (2022).**  
https://arxiv.org/abs/2201.02177

Studies memorization, generalisation, data scale, optimisation and delayed generalisation on algorithmic datasets. Useful for interpreting the training-fit/held-out-transfer gap. It does not prove that more training will fix RE:Reason.

### 2. Grokking Modular Arithmetic
**Andrey Gromov (2023).**  
https://arxiv.org/abs/2301.02679

Analyses modular arithmetic in small fully connected networks and structured feature maps associated with generalising solutions. Useful for identifying periodic/cyclic feature signatures. The model and task presentation differ from RE:Reason's sequential register updates.

### 3. Learning to Grok: Emergence of In-Context Learning and Skill Composition in Modular Arithmetic Tasks
**Tianyu He, Darshil Doshi, Aritra Das, Andrey Gromov (NeurIPS 2024).**  
https://proceedings.neurips.cc/paper_files/paper/2024/hash/17d60fef592086d1a5cb136f1946df59-Abstract-Conference.html

Studies out-of-distribution generalisation across modular arithmetic functions, including effects of task diversity and model depth. Useful for framing held-out combinations as systematic generalisation. It studies transformers and in-context learning, not RE:Reason's recurrent register machine.

### 4. Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product
**Neil Rohit Mallinar et al. (ICML 2025).**  
https://proceedings.mlr.press/v267/mallinar25a.html  
Preprint: https://arxiv.org/abs/2407.20199

Reports modular-arithmetic grokking in Recursive Feature Machines and connects generalisation with task-relevant, block-circulant features. Useful for considering feature learning beyond a specific neural architecture. It does not establish that RFM suits the temporal state-tracking task.

### 5. Neural Algorithmic Reasoning
**Petar Veličković and Charles Blundell (Patterns, 2021).**  
https://arxiv.org/abs/2105.02761  
Open-access journal version: https://pmc.ncbi.nlm.nih.gov/articles/PMC8276006/

A foundational perspective on neural networks that learn to execute algorithmic computations. Useful for positioning RE:Reason within algorithm learning and generalisation.

## Priority 2 — Recurrent computation and algorithmic generalisation

### 6. Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks
**Avi Schwarzschild et al. (NeurIPS 2021).**  
https://proceedings.neurips.cc/paper/2021/hash/3501672ebc68a5524629080e3ef60aef-Abstract.html

Studies recurrent networks that generalise to harder instances by performing additional recurrent computation at inference. Useful for distinguishing sequence-length generalisation from generalisation of a specific arithmetic transition.

### 7. Neural GPUs Learn Algorithms
**Łukasz Kaiser and Ilya Sutskever (2016).**  
https://arxiv.org/abs/1511.08228

Explores neural architectures and training methods for algorithmic generalisation, including arithmetic tasks and longer inputs. Useful as an architectural comparison, not a direct prescription.

### 8. Neural Turing Machines
**Alex Graves, Greg Wayne, Ivo Danihelka (2014).**  
https://arxiv.org/abs/1410.5401

Introduces differentiable external memory and evaluates tasks such as copying, sorting and associative recall. Useful background on structured memory alternatives. It does not imply that RE:Reason needs external memory.

### 9. Evolutionary Training and Abstraction Yields Algorithmic Generalization of Neural Computers
**Daniel Tanneberg, Elmar Rueckert, Jan Peters (Nature Machine Intelligence, 2020).**  
https://www.nature.com/articles/s42256-020-00255-1

Investigates modularity and abstraction in a memory-augmented neural computer, reporting strong generalisation across algorithmic configurations. Useful for generating hypotheses about separating operations from data manipulation; the architecture and training procedure differ from RE:Reason.

## Priority 3 — Variable binding and systematic generalisation

### 10. Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems
**Paul Smolensky (1990).**  
https://doi.org/10.1016/0004-3702(90)90007-M

A foundational formal treatment of representing variable–value bindings and symbolic structure in distributed connectionist systems. Relevant to register IDs, operand values and binding interference. It does not prove that RE:Reason needs tensor-product representations.

### 11. Human-like Systematic Generalization through a Meta-Learning Neural Network
**Brenden M. Lake and Marco Baroni (Nature, 2023).**  
https://doi.org/10.1038/s41586-023-06668-3

Investigates systematic compositional generalisation and a meta-learning approach designed for novel combinations. Useful for distinguishing compositional transfer from interpolation over familiar examples.

## Optional recent mechanistic perspective

### 12. Latent Algorithmic Structure Precedes Grokking: A Mechanistic Study of ReLU MLPs on Modular Arithmetic
**Anand Swaroop (2026 preprint).**  
https://arxiv.org/abs/2603.23784

Examines internal feature structure in modular-arithmetic MLPs and argues that algorithm-like structure can precede successful generalisation. Treat as a recent preprint and independently assess its methods before using it to motivate an intervention.

## Suggested reading order

1. Power et al. — memorization, grokking, data and optimisation.
2. Gromov — modular arithmetic and structured features.
3. He et al. — held-out modular functions and systematic generalisation.
4. Veličković & Blundell — neural algorithmic reasoning.
5. Schwarzschild et al. — recurrent algorithmic generalisation.
6. Mallinar et al. — feature learning beyond one neural architecture.
7. Smolensky — variable-binding foundations.
8. Neural GPUs, Neural Turing Machines and the neural Harvard computer — architectural alternatives only when project evidence motivates them.

## Experimental citation checklist

Before citing a paper to justify an experiment, record:
- The exact claim and the section/figure supporting it.
- Architecture, task, data distribution, objective and evaluation split.
- Which aspects match RE:Reason and which differ.
- Whether the paper motivates a hypothesis, supplies a method, or reports directly comparable evidence.
- What result would falsify the proposed hypothesis.

**Governance reminder:** Literature can motivate a hypothesis or diagnostic; it cannot by itself license a RE:Reason core revision. Use the project's controlled evidence and registered decision criteria.
