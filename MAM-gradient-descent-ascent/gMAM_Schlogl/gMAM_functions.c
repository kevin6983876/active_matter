//Useful functions

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, (void *)0);
  return (double) tv.tv_sec + tv.tv_usec*1e-6;
}


int findArgmax(int size, double* array){
    int c;
    int i;
    c=0;
    for(i=0;i<size;i++){
	if(array[c]<array[i]) c=i;
    }	
    return c;
}

double returnMax(double a, double b){
	if(a>b) return a;
	else return b;
}



void recentreArray(int size, double* array, double* array_copy, int argmax){
    int i;
    for(i=0;i<size;i++){
	array_copy[i] = array[i];	
    }
    for(i=0;i<size;i++){
	array[(i+size/2)%size] = array_copy[(i+argmax)%size];
    }
    return;
}


void sum( double** rho1, double** rho2, double** rho3, int Ncopy, int Nx){
	int i,x;
    for(i=0;i<Ncopy;i++){
		for(x=0;x<Nx;x++){
	    rho3[i][x]= rho1[i][x] + rho2[i][x];
		}
    }
    return;
}

void diff( double** rho1, double** rho2, double** rho3, int Ncopy, int  Nx){
	int i,x;
    for(i=0;i<Ncopy;i++){
		for(x=0;x<Nx;x++){
	    rho3[i][x]= rho1[i][x] - rho2[i][x];
		}
    }
    return;
}


void linearCombi( double a, double** rho1, double b, double** rho2, double** rho3, int Ncopy, int  Nx){
	int i,x;
    for(i=0;i<Ncopy;i++){
		for(x=0;x<Nx;x++){
	    rho3[i][x]= a*rho1[i][x] + b* rho2[i][x];
		}
    }
    return;
}


void multHad( double** rho1, double** rho2, double** rho3, int Ncopy, int  Nx){
	int i,x;
    for(i=0;i<Ncopy;i++){
		for(x=0;x<Nx;x++){
	    rho3[i][x]= rho1[i][x] * rho2[i][x];
		}
    }
    return;
}


void scalarMult(double a, double** rho1, int Ncopy, int  Nx){
	int i,x;
    for(i=0;i<Ncopy;i++){
		for(x=0;x<Nx;x++){
	    rho1[i][x] *= a ;
		}
    }
}



void updateA1fromA2Real( double** rho1, double** rho2, int Ncopy, int  Nx){
	int i,x;
    for(i=0;i<Ncopy;i++){
		for(x=0;x<Nx;x++){
	    rho1[i][x] = rho2[i][x];
		}
    }
    return;
}

void updateA1fromA2RealExceptEndPoints( double** rho1, double** rho2, int Ncopy, int  Nx){
	int i,x;
    for(i=1;i<Ncopy-1;i++){
		for(x=0;x<Nx;x++){
	    rho1[i][x] = rho2[i][x];
		}
    }
    return;
}


void updateA1fromA2Complex( fftw_complex** rho1, fftw_complex** rho2, int Ncopy, int  Nx){
	int i,x;
    for(i=0;i<Ncopy;i++){
		for(x=0;x<Nx;x++){
	    	rho1[i][x] = rho2[i][x];
		}
    }
    return;
}


// Define diagonal (1d-array, 0->N-1) and upper diag (1d-array, 0->N-2)
void solveUpperDiagonalSystem(fftw_complex* diag,  fftw_complex* upperDiag, fftw_complex* x,  fftw_complex* b, int N){
	int i;
	x[N-1] =  b[N-1]/diag[N-1];
	for(i=N-2;i>-1;i--){
		x[i] = (b[i] - upperDiag[i]*x[i+1]) / diag[i] ;
	}
	return;
}

// Define diagonal (1d-array, 0->N-1) and lower diag (1d-array, 1->N-1)
void solveLowerDiagonalSystem( fftw_complex* diag,  fftw_complex* lowerDiag, fftw_complex* x,  fftw_complex* b, int N){
	int i;
	x[0] =  b[0]/diag[0];
	for(i=1;i<N;i++){
		x[i] = (b[i] - lowerDiag[i]*x[i-1]) / diag[i] ;
	}
	return;
}


/* SECOND ORDER diagonal */
// Define diagonal (1d-array, 0->N-1) and upper diag (1d-array, 0->N-2), upper 2nddiag (1d-array, 0->N-3)
void solveUpperDiagonalSystem2order(fftw_complex* diag,  fftw_complex* upperDiag, fftw_complex* upper2ndDiag, fftw_complex* x,  fftw_complex* b, int N){
	int i;
	x[N-1] =  b[N-1]/diag[N-1];
	i = N-2;
	x[i] = ( b[i] - upperDiag[i]*x[i+1] ) / diag[i] ;
	for(i=N-3;i>-1;i--){
		x[i] = ( b[i] - upperDiag[i]*x[i+1] - upper2ndDiag[i]*x[i+2] ) / diag[i] ;
	}
	
	return;
}

// Define diagonal (1d-array, 0->N-1) and lower diag (1d-array, 1->N-1)
void solveLowerDiagonalSystem2order( fftw_complex* diag,  fftw_complex* lowerDiag, fftw_complex* lower2ndDiag, fftw_complex* x,  fftw_complex* b, int N){
	int i;
	x[0] =  b[0]/diag[0];
	i = 1;
	x[i] = (b[i] - lowerDiag[i]*x[i-1]) / diag[i] ;
	for(i=2;i<N;i++){
		x[i] = ( b[i] - lowerDiag[i]*x[i-1] - lower2ndDiag[i]*x[i-2]) / diag[i] ;
	}
	return;
}





double normL2(int Nx, double dx, double* rho1){
    int i;
    double norm=0;
    for(i=0;i<Nx;i++){
 	norm += (rho1[i])*(rho1[i]);
    }
    return sqrt(norm*dx);
}

double distanceL2(int Nx, double dx, double* rho1, double* rho2){
    int i;
    double norm=0;
    for(i=0;i<Nx;i++){
 	norm += (rho1[i]-rho2[i])*(rho1[i]-rho2[i]);
    }
    return sqrt(norm*dx);
}

double computeString(int Ncopy, int Nx, double dx, double* s, double** rho){
    int i;
    double totalLength;
    s[0]=0;
    for(i=1;i<Ncopy;i++){
		s[i] = s[i-1] + distanceL2(Nx, dx, rho[i-1], rho[i]);
    }
    totalLength = s[Ncopy-1];
    for(i=1;i<Ncopy;i++)
		s[i] = s[i]/totalLength;
    return totalLength;
}


void computeLagrangian(double* gMAMLagrangian, double** rho, double** theta, int Ncopy, int Nx, double ds, double dx){
    int i,x;
    for(i=1;i<Ncopy-1;i++){
    	gMAMLagrangian[i] =0;
    	for(x=0;x<Nx;x++){
    	gMAMLagrangian[i] += dx*( (rho[i][x]-rho[(i-1+Ncopy)%Ncopy][x])/ds * theta[i][x] );
		}	
    }
    gMAMLagrangian[0] = 0;
	gMAMLagrangian[Ncopy-1] = 0;
    return;
}


//////////////// Model dependent functions

//Compute non linear part of the force
void FNL(double** NL, double** rho, double** theta , int Ncopy, int Nx, double dx, double k0, double k1,  double k2, double k3) {
    int i;
    int j;

    for(i=0;i<Ncopy;i++){
  		for(j=0;j<Nx; j++){
	    	NL[i][j] = (k0 + k2*rho[i][j]*rho[i][j]) - (k1*rho[i][j] + k3*rho[i][j]*rho[i][j]*rho[i][j])  ;
		}
    }
    //free(rhoProjected);
    return;
}

//Compute non linear part of the force
void computeHamiltonian(double* gMAMHamiltonian, double** rho, double** theta, double** NL_rho, int Ncopy, int Nx, double dx, double D, double k0, double k1, double k2, double k3){
    int i,x;
    double gradTheta;
    for(i=1;i<Ncopy-1;i++){
    	gMAMHamiltonian[i] = 0;

    	for(x=0;x<Nx;x++){
    		gMAMHamiltonian[i] += dx*( 
    			D*rho[i][x]* ( exp(theta[i][(x-1+Nx)%Nx]-theta[i][x]) 
    							+ exp(theta[i][(x+1)%Nx]-theta[i][x]) - 2 )     
    			+ (k0 + k2*rho[i][x]*rho[i][x])*(exp(theta[i][x])-1)
    			+ (k1*rho[i][x] + k3*rho[i][x]*rho[i][x]*rho[i][x])*(exp(-theta[i][x])-1)   			
    			) ;
		}	
    }
    gMAMHamiltonian[0]       = 0;
	gMAMHamiltonian[Ncopy-1] = 0;
    return;
}




/***************************************************************
/***************     Initialization function 
***************************************************************/

void initFields(int Ncopy, int Nx, double** rho, double** theta, double rho0, int upward){
    int i,j;
    
  if(1==upward){
  	  for(j=0;j<Nx;j++){
		  rho[Ncopy-1][j]       =  8.0/5.0  ; 
		  rho[0][j] =   0.5 ;
	  }
	  for(i=1;i<Ncopy-1;i++){
		  if(i>Ncopy/2){
			  for(j=0;j<Nx;j++){
				  rho[i][j] =  8.0/5.0   +  0.5*sin(2.*M_PI*j/Nx); 
			  }
		  }
		  else{
			  for(j=0;j<Nx;j++){
				  rho[i][j] =    0.5   + 0.5*sin(2.*M_PI*j/Nx); 
			  }
		  }
	  }
  }
  else{
	  for(j=0;j<Nx;j++){
		  rho[0][j]  =  8.0/5.0  ; 
		  rho[Ncopy-1][j] =   0.5 ;
	  }
	  for(i=1;i<Ncopy-1;i++){
		  if(i<Ncopy/2){
			  for(j=0;j<Nx;j++){
				  rho[i][j] =  8.0/5.0   +  0.5*sin(2.*M_PI*j/Nx); 
			  }
		  }
		  else{
			  for(j=0;j<Nx;j++){
				  rho[i][j] =    0.5   + 0.5*sin(2.*M_PI*j/Nx); 
			  }
		  }
	  }
  
  }
  
  return;
}




void initFromLastFiles(int Ncopy, int Nx, double** rho, double** theta, double rho0, int upward)
{
	int j;
	int i;
	double x;
	FILE* fichierRho;
	FILE* fichierTheta;
	fichierRho = fopen("rho_Last.txt","r");
	fichierTheta = fopen("theta_Last.txt","r");
	if( (fichierRho!=NULL) && (fichierTheta!=NULL))
	{
		for (i=0;i<Ncopy;i++)
		{
		    for(j=0;j<Nx;j++){
		    	if(Nx==j){
		        	fscanf(fichierRho,"%lf\n", &x);
		        	rho[i][j] = x;
		        }
		    	else{
		    		fscanf(fichierRho,"%lf\t", &x);
		        	rho[i][j] = x;
		    	}
		    }
		}
		fclose(fichierRho);
		for (i=0;i<Ncopy;i++)
		{
		    for(j=0;j<Nx;j++){
		    	if(Nx==j){
		        	fscanf(fichierTheta,"%lf\n", &x);
		        	theta[i][j] = x;
		        }
		    	else{
		    		fscanf(fichierTheta,"%lf\t", &x);
		        	theta[i][j] = x;
		    	}
		    }
		}
		fclose(fichierTheta);
		return;
	}
	else
	{
		printf("No former simulations: initiate from scratch");
		initFields(Ncopy, Nx, rho, theta, rho0, upward);
		return;
	}
}







//**********************************************************************************
//  PRINT FUNCTIONS
///******************
void createListOfParameters(int Ncopy, int Nx, double D, 
			    double dt, int plotStep, int iterations)
{
    char chaine[100];
    FILE* output;
    sprintf(chaine, "listOfParameters.txt");
    output = fopen(chaine,"w");
    
    fprintf(output,"%d\t", Ncopy);
    fprintf(output,"%d\t", Nx);
    fprintf(output,"%lf\t",D);
    fprintf(output,"%lf\t",dt);
    fprintf(output,"%d\t",plotStep);
    fprintf(output,"%d\t",iterations);
	
    fclose(output);
    return;
}


void printRhoTXT(int Ncopy, int Nx, int c, double** rho)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine, "string_rho_%d.txt", c);
    output = fopen(chaine,"w");

    for (i=0;i<Ncopy;i++){
        for(j=0;j<Nx;j++){
            fprintf(output,"%.5e\t",rho[i][j]);
        }
        fprintf(output,"\n");
    }
    fclose(output);
    /*free(output);*/
    return;
}



void printRhoTXTgMAM(int Ncopy, int Nx, int c, double** rho)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine, "gMAM_rho_%d.txt", c);
    output = fopen(chaine,"w");

    for (i=0;i<Ncopy;i++){
        for(j=0;j<Nx;j++){
            fprintf(output,"%.6e\t",rho[i][j]);
        }
        fprintf(output,"\n");
    }
    fclose(output);
    /*free(output);*/
    return;
}


void printThetaTXT(int Ncopy, int Nx, int c, double** theta)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine, "string_theta_%d.txt", c);
    output = fopen(chaine,"w");

    for (i=0;i<Ncopy;i++){
        for(j=0;j<Nx;j++){
            fprintf(output,"%.5e\t", theta[i][j]);
        }
        fprintf(output,"\n");
    }
    fclose(output);
    /*free(output);*/
    return;
}


void printThetaTXTgMAM(int Ncopy, int Nx, int c, double** theta)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine, "gMAM_theta_%d.txt", c);
    output = fopen(chaine,"w");

    for (i=0;i<Ncopy;i++){
        for(j=0;j<Nx;j++){
            fprintf(output,"%.6e\t", theta[i][j]);
        }
        fprintf(output,"\n");
    }
    fclose(output);
    /*free(output);*/
    return;
}

void printNormAlongString(int Ncopy, double* b1)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine, "normAlongString.dat");
    output = fopen(chaine,"a");

    for (i=0;i<Ncopy;i++){
            fprintf(output,"%lf\t", b1[i]);
        }
        fprintf(output,"\n");
    fclose(output);
    /*free(output);*/
    return;
}


void printLambdaGMAM(int Ncopy, int Nx, int c, double* lambda)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine,"gMAMLambda_Ncopy%d_L%d.txt", Ncopy, Nx);
    output = fopen(chaine,"a");

    for(i=0;i<Ncopy-1;i++){
        fprintf(output,"%.5e\t", lambda[i]);
    }
    fprintf(output,"%.5e\n", lambda[Ncopy-1]);
    fclose(output);
    /*free(output);*/
    return;
}




void printHamiltonianGMAM(int Ncopy, int Nx, int c, double* gMAMHamiltonian)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine,"gMAMHamiltonian_Ncopy%d_L%d.txt", Ncopy, Nx);
    output = fopen(chaine,"a");

    for(i=0;i<Ncopy-1;i++){
        fprintf(output,"%.5e\t", gMAMHamiltonian[i]);
    }
    fprintf(output,"%.5e\n", gMAMHamiltonian[Ncopy-1]);
    fclose(output);
    /*free(output);*/
    return;
}

void printLagrangianGMAM(int Ncopy, int Nx, int c, double* gMAMLagrangian)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine, "gMAMLagrangian_Ncopy%d_L%d.txt", Ncopy, Nx);
    output = fopen(chaine,"a");

    for(i=0;i<Ncopy-1;i++){
        fprintf(output,"%.5e\t", gMAMLagrangian[i]);
    }
    fprintf(output,"%.5e\n", gMAMLagrangian[Ncopy-1]);
    
    fclose(output);
    /*free(output);*/
    return;
}


void printLagrangianString(int Ncopy, int Nx, int c, double* gMAMLagrangian)
{
    char chaine[100];
    int i;
    int j;
    FILE* output;
    sprintf(chaine, "stringLagrangian_Ncopy%d_L%d.txt", Ncopy, Nx);
    output = fopen(chaine,"a");

    for(i=0;i<Ncopy-1;i++){
        fprintf(output,"%.5e\t", gMAMLagrangian[i]);
    }
    fprintf(output,"%.5e\n", gMAMLagrangian[Ncopy-1]);
    
    fclose(output);
    /*free(output);*/
    return;
}












