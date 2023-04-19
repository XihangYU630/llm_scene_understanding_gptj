from ff_train import *

t_acc = None
t_loss = None 

v_acc = None
v_loss = None 

test_acc = None
test_loss = None 
with open("/home/ruoyuw/Desktop/487/llm_scene_understanding/ff_results/train_acc.pkl",
                          "rb") as fp:
    t_acc = pickle.load(fp)



with open("/home/ruoyuw/Desktop/487/llm_scene_understanding/ff_results/train_losses.pkl",
                          "rb") as fp:
    t_loss = pickle.load(fp)

with open("/home/ruoyuw/Desktop/487/llm_scene_understanding/ff_results/val_acc.pkl",
                          "rb") as fp:
    v_acc = pickle.load(fp)



with open("/home/ruoyuw/Desktop/487/llm_scene_understanding/ff_results/val_losses.pkl",
                          "rb") as fp:
    v_loss = pickle.load(fp)
    
    
with open("/home/ruoyuw/Desktop/487/llm_scene_understanding/ff_results/test_acc.pkl",
                          "rb") as fp:
    test_acc = pickle.load(fp)



with open("/home/ruoyuw/Desktop/487/llm_scene_understanding/ff_results/test_loss.pkl",
                          "rb") as fp:
    test_loss = pickle.load(fp)


# Training loss
Order = ['nyuClass_GT', 'nyuClass_Proxy', 'mpcat40_GT', 'mpcat40_Proxy']

#os.makedirs('R_GRAPH')
plt.figure()

# Plot train loss
for i in range(2):
    c_loss = t_loss[i]
    plt.plot(c_loss, label = 'Train_' +Order[i])
# Plot val loss
for i in range(2):
    c_loss = v_loss[i]
    plt.plot(c_loss, label = 'Val_' + Order[i])
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(0.55, 0.5), fontsize=20)
plt.title('Training loss for nyuClass', fontsize=20)
fig = plt.gcf()

fig.set_size_inches(10, 10)
plt.savefig('./R_GRAPH/nyu_Loss.png')
plt.close()

plt.figure()

# Plot train loss
for i in range(2,4):
    c_loss = t_loss[i]
    plt.plot(c_loss, label = 'Train_' +Order[i])
# Plot val loss
for i in range(2,4):
    c_loss = v_loss[i]
    plt.plot(c_loss, label = 'Val_' + Order[i])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='center left', bbox_to_anchor=(0.85, 0.5))
plt.title('Training loss for mpcat40')
fig = plt.gcf()

fig.set_size_inches(10, 8)
plt.savefig('./R_GRAPH/mpcat40_Loss.png')
plt.close()

c_loss = None

# Plot train accuracy
for i in range(2):
    c_acc = t_acc[i]
    plt.plot(c_acc, label = 'Train_' +Order[i])
# Plot val loss
for i in range(2):
    c_acc = v_acc[i]
    plt.plot(c_acc, label = 'Val_' + Order[i])
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(0.55, 0.75), fontsize=20)
plt.title('Training accuracy for nyuClass', fontsize=20)
fig = plt.gcf()

fig.set_size_inches(10, 8)

plt.savefig('./R_GRAPH/nyu_accuracy.png')
plt.close()

plt.figure()

# Plot train accuracy
for i in range(2,4):
    c_acc = t_acc[i]
    plt.plot(c_acc, label = 'Train_' +Order[i])
# Plot val loss
for i in range(2,4):
    c_acc = v_acc[i]
    plt.plot(c_acc, label = 'Val_' + Order[i])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='center left', bbox_to_anchor=(0.85, 0.5))
plt.title('Training accuracy for mpcat40')
fig = plt.gcf()

fig.set_size_inches(10, 8)
plt.savefig('./R_GRAPH/mpcat40_accuracy.png')
plt.close()



total_acc = []
for i in range(2):
    c_acc = test_acc[i]
    val = torch.mean(torch.tensor(c_acc))
    total_acc.append(val)


plt.figure()
fig = plt.bar(Order[:2], total_acc, color=['tab:blue', 'tab:orange'])
plt.ylim(0, 1)



plt.bar_label(fig, label=total_acc, label_type='edge')


plt.ylabel('Accuracy')
plt.title('Test accuracy')
plt.savefig('./R_GRAPH/test_acc.png')
plt.close()