
```python
"""
训练过程的伪代码
数据和模型准备阶段已省略
编辑验证、测试阶段已省略
run函数是训练的入口函数
model是LM
net是超网络
"""

    def run(
        train_loader: DataLoader,
        valid_loader: DataLoader
    ):
        # 总取样轨迹数为1时效果就已经很好
        for 每个轨迹 in 总取样轨迹数:
            train(train_loader) # train_loader有20批100条的知识
            reset_LM() # 重置LM为编辑前的LM
            valid(valid_loader) # 每个轨迹后验证超网络编辑能力
            reset_LM() # 重置LM为编辑前的LM


    def train(train_loader: DataLoader):

        for tuples in train_loader:
            # train_loader里面有随机取样的20批知识，每批有100条
            # tuples就是其中一批知识

            # 将本批知识喂给LLM，获取LLM微调梯度
            # 并将梯度分解存储到本地(防止显存占用过多)
            get_cache(tuples) 

            # 用超网络预测LM参数变化量。
            # 超网络输入是本地存储的分解后的梯度，输出是LM参数变化量
            param_shifts = predict_param_shifts() 
            
            # 将超网络生成的变化量应用到LM上
            edit_model(param_shifts) 

            for t in tuples["edit_knowledge"]:
                # 对于本批知识中每一条知识，都经过LM并获取loss
                logits = model(**t)["logits"]
                loss = cross_entropy(logits, t["labels"])
                tot_edit_loss += loss
            tot_edit_loss.backward()
            # 注：这里loss是在LM上反向传播的

            # 下面是相关知识和无关知识的计算
            # malmen中将相关与无关知识的loss合称为meta loss
            for t in tuples["equal_knowledge"]:
                # 对于本批知识中每一条的相关知识，都经过LM并获取loss
                logits = model(**t)["logits"]
                loss = cross_entropy(logits, t["labels"])
                tot_gen_loss += loss
            tot_gen_loss.backward()
            # 注：这里loss是在LM上反向传播的

            for t in tuples["unrel_knowledge"]:
                # 对于本批知识中每一条的无关知识，都经过LM并获取loss
                logits = model(**t)["logits"]
                loss = kl_div(
                    refer_logits, # 编辑前LM的logits
                    logits, # 编辑后LM的logits
                    t["labels"]
                ) # 通过编辑前后KL散度来算loss
                tot_loc_loss += loss
            tot_loc_loss.backward()
            # 注：这里loss是在LM上反向传播的

            # 计算meta gradient并在超网络上进行反向传播
            # 这里的参考了Malmen的实现，具体见Malmen的论文
            # meta gradient是通过上面代码中积累的LM上的梯度来计算的
            calculate_metagra(param_shifts)

        # 在20批知识反向传播完后，对超网络进行一次总的参数更新
        optimizer.step()

