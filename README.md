# IC_MF_RL_CW
MATH70120 Advanced Machine Learning Coursework

## 中文说明（之后可删）：
requirements.txt给定了训练的环境，请记得参考。

代码主要分为三块，一块是环境（这一部分我沿用了源码并做了修改，包括envs，schedules，segment_tree, replay_buffer和utils五个文件），一块是算法及参数（除了复现ddpg，我添加了sac，q_learning本来想加但是它是离散策略可能不适用于我们的情况，你们如果愿意的话也可以实现试一下，因为yufei提了一嘴），一块是训练与测试（见train和test，sac算法与train放在一个文件里）。experimentmanager这个文件是用来管理实验过程的，运行后会把时间、参数、权重存在对应的文件夹里，你们跑一次就会明白了。

具体算法参数我放在config.py的文件里，但不一定全，比如你们也可以练模型的时候调整网络结构的参数（我并没有放在config里），具体训练多少次，多少次输出一次，你们也自己酌情调整，我应该有加备注。有问题随时dd我。

实现细节还挺多的，比如price有两种，一个是基于gbm的，一个是基于sabr的，包括delta源码也给出了两个算法，一个是普通的delta，一个是bartlett delta，论文应该是对比了这两种和rl练出来的策略的不同，如果你们觉得我理解的有问题也记得dd我。

