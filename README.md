# understanding_knowledge_distillation
Project submitted as part of the Term paper for Deep Learning Applications- Course in the Data Science Master at the Barcelona School of Economics. \
The aim of this experiment is to succesfully perform knowledge distillation over long training schedules and compare the performance of this novel technique against benchmarks.
The main training loop can be found in main_training.py, with the specialized KL loss function designed to compare the outputs of a teacher and student model, while main.py served as a config changer to execute the experiments as needed.
