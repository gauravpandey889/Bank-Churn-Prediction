def engineer(df):
    x=df.drop(columns=['Exited','CustomerId','Surname','Year'],errors='ignore').copy()
    x['BalanceSalaryRatio']=x['Balance']/(x['EstimatedSalary'].clip(lower=1))
    x['EngagementProducts']=x['IsActiveMember']*x['NumOfProducts']
    x['AgeTenure']=x['Age']*x['Tenure']
    x['ProductsPerTenure']=x['NumOfProducts']/(x['Tenure']+1)
    return x
