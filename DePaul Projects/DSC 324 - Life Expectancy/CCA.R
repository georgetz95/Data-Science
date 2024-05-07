library(foreign)
library(CCA)
library(yacca)
library(MASS)

ccaWilks = function(set1, set2, cca)
{
  ev = ((1 - cca$cor^2))
  ev
  
  n = dim(set1)[1]
  p = length(set1)
  q = length(set2)
  k = min(p, q)
  m = n - 3/2 - (p + q)/2
  m
  
  w = rev(cumprod(rev(ev)))
  
  # initialize
  d1 = d2 = f = vector("numeric", k)
  
  for (i in 1:k) 
  {
    s = sqrt((p^2 * q^2 - 4)/(p^2 + q^2 - 5))
    si = 1/s
    d1[i] = p * q
    d2[i] = m * s - p * q/2 + 1
    r = (1 - w[i]^si)/w[i]^si
    f[i] = r * d2[i]/d1[i]
    p = p - 1
    q = q - 1
  }
  
  pv = pf(f, d1, d2, lower.tail = FALSE)
  dmat = cbind(WilksL = w, F = f, df1 = d1, df2 = d2, p = pv)
}

economic_features <- c('Income.composition.of.resources', 'Schooling', 'GDP', 'Total.expenditure')
health_features <- c('Life.expectancy', 'Adult.Mortality', 'infant.deaths', 'thinness..1.19.years', 'thinness.5.9.years')

economic <- df[, economic_features]
mortality <- df[, health_features]

head(economic)
head(mortality)

c = matcor(economic, mortality)

cc_mm = cc(economic, mortality)
wilks_mm = ccaWilks(economic, mortality, cc_mm)
round(wilks_mm, 2)

options(scipen=999)
cc = cca(economic,mortality, xscale=TRUE, yscale=TRUE)
summary(cc)

#CV1
helio.plot(cc, cv=1, x.name="Economic Values", 
           y.name="Mortality Values")

helio.plot(cc, cv=2, x.name="Economic Values", 
           y.name="Mortality Values")