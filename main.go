package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func main() {
	data := map[string][]float64{
		"interest_rate":     {2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75},
		"unemployment_rate": {5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1, 6.1, 5.9, 6.2, 6.2, 6.1},
		"index_price":       {1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958, 971, 949, 884, 866, 876, 822, 704, 719},
	}

	// let x be the matrix with interest_rate and unemployment_rate
	interestRates :=data["interest_rate"]
	unemploymentRates := data["unemployment_rate"]

	// Create the matrix with interest_rate and unemployment_rate
	x := mat.NewDense(len(interestRates), 3, nil)
	for i := 0; i < len(interestRates); i++ {
		x.Set(i, 0, 1)
		x.Set(i, 1, interestRates[i])
		x.Set(i, 2, unemploymentRates[i])
	}

	y := mat.NewDense(len(data["index_price"]), 1, data["index_price"])

	fmt.Println(mat.Formatted(x, mat.Prefix(" "), mat.Excerpt(0)))
	fmt.Println(mat.Formatted(y, mat.Prefix(" "), mat.Excerpt(0)))
	// calculate the matrix Transpose
	xT := mat.DenseCopyOf(x.T())
	// calculate the matrix xT * x
	xTx := mat.NewDense(3, 3, nil)
	xTx.Product(xT, x)
	fmt.Println(mat.Formatted(xTx, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the inverse of xTx
	xTxInv := mat.NewDense(3, 3, nil)
	xTxInv.Inverse(xTx)
	fmt.Println(mat.Formatted(xTxInv, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the matrix xT * y
	xTy := mat.NewDense(3, 1, nil)
	xTy.Product(xT, y)
	fmt.Println(mat.Formatted(xTy, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the matrix beta
	beta := mat.NewDense(3, 1, nil)
	beta.Product(xTxInv, xTy)
	fmt.Println(mat.Formatted(beta, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the matrix yHat
	yHat := mat.NewDense(len(data["index_price"]), 1, nil)
	yHat.Product(x, beta)
	fmt.Println(mat.Formatted(yHat, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the matrix residuals
	residuals := mat.NewDense(len(data["index_price"]), 1, nil)
	residuals.Sub(y, yHat)
	fmt.Println(mat.Formatted(residuals, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the σ^2 which is the variance of the residuals
	// turn residuals into a mat.Vector
	residualsVec := mat.NewVecDense(len(data["index_price"]), nil)
	for i := 0; i < len(data["index_price"]); i++ {
		residualsVec.SetVec(i, residuals.At(i, 0))
	}
	// calculate the σ^2 which is the variance of the residualsVec
	variance := mat.Dot(residualsVec, residualsVec) / float64(len(data["index_price"])-3)
	fmt.Println(variance)
	// calculate The standard error of the estimated parameters
	// calculate the matrix (xTx)^-1 * σ^2

	xTxInvSigma2 := mat.NewDense(3, 3, nil)
	xTxInvSigma2.Scale(variance, xTxInv)
	// calculate the diagonal of the matrix xTxInvSigma2
	dia := make([]float64, 3)
	for i := 0; i < 3; i++ {
		dia[i] = xTxInvSigma2.At(i, i)
	}

	// calculate the standard error of the estimated parameters
	standardError := mat.NewDense(3, 1, dia)
	for i := 0; i < 3; i++ {
		standardError.Set(i, 0, math.Sqrt(standardError.At(i, 0)))
	}

	fmt.Println(mat.Formatted(standardError, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the t-statistic
	tStatistic := mat.NewDense(3, 1, nil)

	for i := 0; i < 3; i++ {
		tStatistic.Set(i, 0, beta.At(i, 0)/standardError.At(i, 0))
	}

	fmt.Println(mat.Formatted(tStatistic, mat.Prefix(" "), mat.Excerpt(0)))

	// calculate the p-value
	// calculate the degrees of freedom
	degreesOfFreedom := float64(len(data["index_price"]) - 3)
	// calculate the p-value
	pValue := mat.NewDense(3, 1, nil)
	for i := 0; i < 3; i++ {
		dist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: degreesOfFreedom}
		pValue.Set(i, 0, 2*(1-dist.CDF(math.Abs(tStatistic.At(i, 0)))))
	}

	fmt.Println(mat.Formatted(pValue, mat.Prefix(" "), mat.Excerpt(0)))



}
