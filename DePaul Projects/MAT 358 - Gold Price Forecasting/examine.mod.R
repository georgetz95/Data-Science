examine.mod <- function(mod.fit.obj, p, d, q, P=0, D=0, Q=0, S=1, lag.max=24, savefig=FALSE) {
  if (savefig != FALSE) {
    png(savefig, width=800, height=600, res=100)
  }
	dev.new(width=8, height=6)
	par(mfrow=c(2,1))
	pacf(mod.fit.obj$fit$residuals, main="PACF of Residuals", lag.max)
	if ((P==0)&(D==0)&(Q==0)) {
		title(paste("Model: (", p, ",", d, ",", q, ")", sep=""), adj=0, cex.main=0.75)
	}
	else {
		title(paste("Model: (", p, ",", d, ",", q, ") (", P, ",", D, ",", Q, ") [", S, "]", sep=""), adj=0, cex.main=0.75)
	}
	
	std.resid <- mod.fit.obj$fit$residuals/sqrt(mod.fit.obj$fit$sigma2)
	hist(std.resid, main="Histogram of Standardized Residuals", xlab="Standardized Residuals", freq=FALSE)
	curve(expr=dnorm(x, mean=mean(std.resid), sd=sd(std.resid)), col="red", add=TRUE)
	if (savefig != FALSE) {
	  dev.off()
	}
}
