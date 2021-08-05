import Pkg;
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("Lathe")
Pkg.add("GLM")
Pkg.add("StatsPlots")
Pkg.add("MLBase")
Pkg.add("Missings")
Pkg.add("Statistics")

using DataFrames
using CSV
using Plots
using Lathe
using GLM
using Statistics
using StatsPlots
using MLBase
using Missings

using CSV,DataFrames
df = DataFrame(CSV.File("C:\\Users\\ardwived\\Downloads\\nystocks.csv"))


#Summary Statistics
describe(df)

#Correlation Analysis
scatter(df.open,df.close,xlabel="Opening Price",ylabel="Closing Price")

#Distribution Analysis
density(df.close,title="Density Plot", ylabel="open",xlabel="close",legend=true)

#Split data
using Lathe.preprocess:TrainTestSplit
train,test=TrainTestSplit(df,.75)

#..........................................linear regression Model..............................................................
#GLM Generalized linear Model
using GLM
fm =@formula(open~close)
linreg = lm(fm,train)

#R squared Value
r2(linreg)

#Prediction
test_pred=predict(linreg,test)
train_pred=predict(linreg,train)

#Mesuring Performance (for test)
perf_test=df_original=DataFrame(y_original=test[!,:open],y_pred=test_pred)
perf_test.error=perf_test[!,:y_original]-perf_test[!,:y_pred]
perf_test.error_sq=perf_test.error.*perf_test.error

#Mesuring Performance (for train)
perf_train=df_original=DataFrame(y_original=train[!,:open],y_pred=train_pred)
perf_train.error=perf_train[!,:y_original]-perf_train[!,:y_pred]
perf_train.error_sq=perf_train.error.*perf_train.error

#Loss Function
#MAPE
function mape(perf_df)
    mape=mean(abs.(perf_df.error./perf_df.y_original))
    return
end

#RMSE
function rmse(perf_df)
    rmse=sqrt(mean(perf_df.error.*perf_df.error))
    return rmse
end

#Now printout the values of MAPE and RMSE and visualize them in form of Histogram, to understand didtribution better
#1) test
println("Mean Absolute test error:",mean(abs.(perf_test.error)),"\n")
println("Mean Absolute Percentage test error:",mape(perf_test),"\n")
println("Root Mean test error:",rmse(perf_test),"\n")
println("Mean Square test error:",mean(perf_test.error_sq),"\n")

#histogram
#1)test
histogram(df.open,bins=50,title="Test Error Analysis",ylabel="Frequency",xlabel="Error",legend=true)

#1) train
println("Mean Absolute train error:",mean(abs.(perf_train.error)),"\n")
println("Mean Absolute Percentage train error:",mape(perf_train),"\n")
println("Root Mean train error:",rmse(perf_train),"\n")
println("Mean Square train error:",mean(perf_train.error_sq),"\n")

#histogram
#1)train
histogram(df.close,bins=50,title="Train Error Analysis",ylabel="Frequency",xlabel="Error",legend=true)

#Calculating Accuracy
#Cross Validation Method
function cross_validation(train,k,fm=@formula(open~close))
    a=collect(Kfold(size(train)[1],k))
    for i in 1:k
        row=a[i]
        temp_train=train[row,:]
        temp_test=train[setdiff(1:end,row),:]
        linreg=lm(fm,temp_train)
        perf_test=df_original=DataFrame(y_original=temp_test[!,:open],y_pred=predict(linreg,temp_test))
        perf_test.error=perf_test[!,:y_original]
        perf_test[!,:y_pred]
            println("Mean Error for set $i:",mean(abs.(perf_test.error)))
    end
end

cross_validation(train,10)

#....................................Multiple Linear regression.................................................................

fm1 =@formula(open~close+high+low)
linreg1 = lm(fm1,train)

#R squared Value
r2(linreg1)

#Prediction
test_pred1=predict(linreg1,test)
train_pred1=predict(linreg1,train)

#Mesuring Performance (for test)
perf_test1=df_original1=DataFrame(y_original1=test[!,:open],y_pred1=test_pred1)
perf_test1.error=perf_test1[!,:y_original1]-perf_test1[!,:y_pred1]
perf_test1.error_sq=perf_test1.error.*perf_test1.error;

#Mesuring Performance (for train)
perf_train1=df_original1=DataFrame(y_original1=train[!,:open],y_pred1=train_pred1)
perf_train1.error=perf_train1[!,:y_original1]-perf_train1[!,:y_pred1]
perf_train1.error_sq=perf_train1.error.*perf_train1.error;

#Now printout the values of MAPE and RMSE and visualize them in form of Histogram, to understand didtribution better
#1) test
println("Mean Absolute test error:",mean(abs.(perf_test1.error)),"\n")


#histogram
#1) test
histogram(perf_test1.error,bins=50,title="Test Error Analysis",ylabel="Frequency",xlabel="Error",legend=true)

#2) train
println("Mean Absolute train error:",mean(abs.(perf_train1.error)),"\n")


#histogram
#2) train
histogram(perf_train1.error,bins=50,title="Train Error Analysis",ylabel="Frequency",xlabel="Error",legend=true)

#Cross validation
cross_validation(train,10,fm1)


