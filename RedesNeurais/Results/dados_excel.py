import pandas as pd

df_double = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results_double.csv")
df_conv = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results_conv.csv")
df_dpasp = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results_dpasp.csv")
df_conv2 = pd.read_csv("/home/anapaulatf/Docs/RedesNeurais/Results/results2_conv.csv")


with pd.ExcelWriter('/home/anapaulatf/Docs/RedesNeurais/Results/graficos.xlsx') as writer:  
    df_double.to_excel(writer, sheet_name='soma_simples')
    df_conv.to_excel(writer, sheet_name='convolucao')
    df_dpasp.to_excel(writer, sheet_name='dpasp')
    df_conv2.to_excel(writer, sheet_name='convo2')

