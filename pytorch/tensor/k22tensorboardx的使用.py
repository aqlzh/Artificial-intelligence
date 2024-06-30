from tensorboardX import SummaryWriter


writer = SummaryWriter('../笔记.md')

for i in range(100):
    writer.add_scalar("a",i,global_step=i)
    writer.add_scalar("b",i*i,global_step=i)

writer.close()

# 查看图像命令
# 文件路径不能用中文
# tensorboard --logdir ./