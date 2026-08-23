import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


class Kalman_2D:
    def __init__(self,initx,inity,q_frac=1e-3):
        ols_model = sm.OLS(inity,sm.add_constant(initx)).fit()

        self.alpha = ols_model.params[0]
        self.beta = ols_model.params[1]

        self.theta = np.array([[self.alpha],[self.beta]])
        self.p = ols_model.cov_params()

        self.R = np.array([[ols_model.mse_resid]])   # measurement noise = warmup residual variance (auto-scales to log units)
        
        delta_alpha = self.p[0,0] * q_frac
        delta_beta = self.p[1,1] * q_frac
        self.Q = np.array([[delta_alpha, 0], [0, delta_beta]])
        
        # Do S*S.T = original matrix prevent negative number from rounding the precision
        
        self.s = np.linalg.cholesky(self.p)
        self.s_q = np.linalg.cholesky(self.Q)
        self.s_R = np.linalg.cholesky(self.R)
        
        self.I = np.eye(2)


    @classmethod
    def tune_q_frac(cls,logx,logy,warmup_len):
        candidates = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        results = []

        for q_frac in candidates:
            kalman = cls(logx[:warmup_len],logy[:warmup_len],q_frac=q_frac)
            ets = []
            betas = []

            for x,y in zip(logx[warmup_len:],logy[warmup_len:]):
                _, beta, et = kalman.update(x,y)
                ets.append(et)
                betas.append(beta)

            ljung_box = acorr_ljungbox(ets,lags=[20],return_df=True)
            p_value = float(ljung_box["lb_pvalue"].iloc[0])
            results.append((q_frac,p_value))

        passing = [result for result in results if result[1] > 0.05]
        selected = passing[0] if passing else max(results,key=lambda result: result[1])
        return selected


    def update(self,x,y):
        if np.isnan(x) or np.isnan(y):
            return self.alpha, self.beta, 0.0, 1.0
        else:
            ht = np.array([[1,x]])
            M = np.vstack((self.s.T, self.s_q.T))
            # qr decomposition
            _, r = np.linalg.qr(M)
            self.s = r.T
            
            M_meas = np.block([[self.s_R.T, np.zeros((1,2))],
                                [self.s.T @ ht.T,  self.s.T]])
            
            _, r_meas = np.linalg.qr(M_meas)
            x_val = r_meas[0,0]
            y_val = r_meas[0:1,1:3]
            z_val = r_meas[1:3,1:3]
            
            kt = y_val.T/x_val
            et = y - (ht @ self.theta)
            
            self.theta = self.theta + (kt @ et)    
            self.s = z_val.T

            et = et.item()
            # z-score = et / abs(x_val).item()
            
            alpha_now = self.theta[0].item()
            beta_now = self.theta[1].item()

        return alpha_now, beta_now, et   




        
