package main

import (
	"fmt"
	ols "wonderstone/MLR/OLS"
)

func main() {
	fmt.Println("Hello, World!")
	// read the csv file
	X, Y := ols.ReadCSV("./OLS/data.csv", true, "index_price", "interest_rate", "unemployment_rate")
	// Run the OLS model
	olsRes := ols.NewOLS(X, Y)
	olsRes.Run()
	// print the Coefficients 
	fmt.Println("Coefficients:")
	co:=olsRes.Coefficients().RawMatrix().Data
	// print the raw data only
	fmt.Println(co)

}
