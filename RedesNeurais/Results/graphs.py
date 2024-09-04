import pandas as pd
import matplotlib.pyplot as plt

df_double = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results_double.csv")
df_conv = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results_conv.csv")
df_dpasp = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results_dpasp.csv")
df_conv2 = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results2_conv.csv")

# COMPARATIVO ENTRE AS DUAS REDES
x = range(1,14)
# Plotando os resultados
plt.figure(figsize=(12, 5))
# Loss
plt.subplot(1, 2, 1)
plt.plot(x, df_double['train_losses_double'], label='No domain knowledge')
plt.plot(x, df_conv['train_losses_conv'], label='With domain knowledge')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss for Epoch')

# Acurácia
plt.subplot(1, 2, 2)
plt.plot(x, df_double['test_accuracies_double']*100, label='No domain knowledge')
plt.plot(x, df_conv['test_accuracies']*100, label='With domain knowledge')        
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy for Epoch')
plt.suptitle("Metrics - learning rate 0.15e-3")
plt.savefig('RedesNeurais/Results/graph_dois_casos.png')

# COMPARATIVO DPASP
plt.figure(figsize=(12, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(x, df_conv['test_accuracies']*100, label='Neural Network')
plt.plot(x, df_dpasp['Accuracy']*100, label='dPASP')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy comparison between NN and dPASP')  

# Tempo
plt.subplot(1, 2, 2)
plt.plot(x, df_conv2["cumulative_time"], label='Neural Network')
plt.plot(x, df_dpasp["Tempo"], label='dPASP')
plt.xlabel('Epoch')
plt.ylabel('Time(s)')
plt.legend()
plt.title('Time comparison between NN and dPASP')  

plt.savefig('RedesNeurais/Results/graph2_NN_dpasp.png')



