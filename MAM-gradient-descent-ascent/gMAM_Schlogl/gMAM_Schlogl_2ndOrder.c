#include "complex.h"
#include <fftw3.h>
#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include "time.h"
#include "gMAM_functions.c"

#define M_PI 3.14159265358979323846

/***************************************
Code that finds the string between metastable states. 
We have implemented the string method.
 compile with: gcc file.c -lfftw3 -lm -O3    
********************************************/


  /*            General comment: ALGORITHM of SEMI EXPLICIT SOLVING
   *
   * the principle is the following : we are going to compute the space derivative in Fourier space
   * we will also perform the time evolution on the complexe field, in Fourier space.
   * we have to compute the non linear parts in real space though. Thus the algorithm is the following : 
 
   * loop (c <total_iterations):
   * 1- compute the non linear parts
   * 2- Fourier transform them
   * 3- do the time evolution step
   * 4- do the backward Fourier transform
   * 5- c=c+1
   * end loop
   *
   *
   * */





/***************************************************************/
/*****************************  MAIN  **************************/
/***************************************************************/



int main(int argc, char *argv[]){
  /*
     argc is the number of argument of the command line, and argv is a list
     of string containing the arguments. argv[0] is the name of the executable.
  */

 char params[] = "Input: Ncopy Nx dt zeta plotStep iterations D upward";

    // Check that the number of inputs is correct
   	if(argc!=9){ //should match: number of parameters + 1 (.exe)
     printf("%s\n",params);
     exit(1);
  	}




    double timeRun,start,stop;
    
    start = get_time();
    
    /*numerical parameters */
    double zeta; //implicitness
    int i;
    int j;
    int c;

    int lastPoint;
    
    int Ncopy;
    double dx; //space discretization 
    double dt; //time discretization 
    
    double dtau; //to solve theta at each step
  	double error;
  
    int plotStep;
    int iterations;
    
    /*operators for time evolution*/
    fftw_complex* Linop_diffusion; 

	
    /*numerical variables*/
    int x;   //iterator for real space
    int k;  //iterator for complexe space (fourier space )
    int kx;  //iterator for complexe space (fourier space )
    int argmax; //localize max of an array

    /*observables*/
    // 2-D arrays
    double** rho; //density field in real space
	double** theta; //noise field in real space
	double** thetaBis;
	
    double** rhoBis; //copy density field in real space
    double** NL_rho; //non linear part in real space for equation on rho
    double** determinFlow;
    
    fftw_complex** rhoFourier; //density field in Fourier space
    fftw_complex** NL_rhoFourier; //non linear part in Fourier space for equation on rho
	
	// 1-D arrays
  	double* convenientArray;
  	
  	double* lambdaTime;
    double* gMAMLagrangian;
  	double* gMAMHamiltonian;
  
    double* s; //size Ncopy: string array : Ncopy
    double* normalizedString;
  
    double q; //barycentre
	
	/* variables to compute the string*/
	double ds;
	double stringLength;


    /*physical parameters*/
    int Nx; //number of discrte sites
    
    double D; //diffusion coefficient
	
	double k0,k1,k2,k3;
	k0 = 0.8;
	k1 = 2.9;
	k2 = 3.1;
	k3 = 1.;
	
  	int upward;
  
    /*recording stuff*/
    FILE *output_param    ; // File where parameters are written
    FILE *output_data     ; // File where data are written
    char   filename[80]   ; // string in which filenames are written


    /*plan used to perform the fourier transform*/
    fftw_plan rho_r2c; // Transform real 1d field to into complex 1d
    fftw_plan rho_c2r; // Transform complex 1d field into real 1d field


    /*initialisation*/
    
	// READING THE INPUT: 12 parameters + executable
	i = 1;

	Ncopy	   = strtod(argv[i], NULL); i++; //typically 100 or more
	Nx   	   = strtod(argv[i], NULL); i++; //typically 100 or more
	dt         = strtod(argv[i], NULL); i++;  //typically 0.005
	zeta       = strtod(argv[i], NULL); i++; //0 explicit, 1 implicit, 0.5 Cranck-Nicolson for Linop
	plotStep    = strtod(argv[i], NULL); i++;  
	iterations  = strtod(argv[i], NULL); i++;
	D           = strtod(argv[i], NULL); i++;//diffusion
	upward      = strtod(argv[i], NULL); i++;//1 if upward
	
	double rho0;
	rho0   =  0.7;

    int Niterations_string = iterations;
    int Niterations_gMAM   = iterations*100;
	
    
    // numerical parameter:
    dx = (double) 1.;
    ds = (double) 1/ (double) (Ncopy-1);
  
    //printf("dx %lf\n", dx );
  
    createListOfParameters( Ncopy, Nx,  D, dt,  plotStep,  iterations);
  
    //ARRAY allocation 
    // 2-D arrays
    rho       = fftw_malloc(sizeof(double*) * Ncopy); 
	rhoBis    = fftw_malloc(sizeof(double*) * Ncopy); 
  	theta     = fftw_malloc(sizeof(double*) * Ncopy); 
  	thetaBis  = fftw_malloc(sizeof(double*) * Ncopy); 
  	
    NL_rho       = fftw_malloc(sizeof(double*) * Ncopy); 
    determinFlow = fftw_malloc(sizeof(double*) * Ncopy); 
    
    rhoFourier    = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
    NL_rhoFourier = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 


    
    // 1-D arrays 
    s                = fftw_malloc(sizeof(double)* Ncopy); 
    lambdaTime       = fftw_malloc(sizeof(double)* Ncopy); 
    normalizedString = fftw_malloc(sizeof(double)* Ncopy); 
 	convenientArray  = fftw_malloc(sizeof(double)* Ncopy); 
 	gMAMLagrangian   = fftw_malloc(sizeof(double)* Ncopy); 
  	gMAMHamiltonian  = fftw_malloc(sizeof(double)* Ncopy); 
 	
 	
    //complex arrays	
    rhoFourier    = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
    NL_rhoFourier = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
 
	
    Linop_diffusion      = fftw_malloc(sizeof(fftw_complex) * Nx);
  
    for (i = 0; i < Ncopy; i++)
    {
        rho[i]    = fftw_malloc(sizeof(double) * Nx);
		rhoBis[i] = fftw_malloc(sizeof(double) * Nx);
        theta[i]    = fftw_malloc(sizeof(double) * Nx); 
        thetaBis[i] = fftw_malloc(sizeof(double) * Nx); 
        
        NL_rho[i]     = fftw_malloc(sizeof(double) * Nx);
        rhoFourier[i] = fftw_malloc(sizeof(fftw_complex) * Nx); 
		NL_rhoFourier[i] = fftw_malloc(sizeof(fftw_complex) * Nx);  
	
		determinFlow[i] = fftw_malloc(sizeof(double) * Nx);
		lambdaTime[i]   = 0;
		normalizedString[i] = (double) i/ (double) (Ncopy-1);
    }
    
    
   
    //Declare plans
    //real to complex
    rho_r2c = fftw_plan_dft_r2c_1d(Nx, rho[0], rhoFourier[0], FFTW_MEASURE);

    //complex to real 
    rho_c2r = fftw_plan_dft_c2r_1d(Nx, rhoFourier[0], rho[0], FFTW_MEASURE);

	double kp_x;
	double fcos;
	double ksin;

	for(k=0;k<Nx;k++)
	{                      
	  kp_x = 2*M_PI*(double)k/(double)Nx;
	  fcos = 2*(cos(kp_x)-1.)/(dx*dx);//Centered laplacian
	  ksin = sin(kp_x)/(dx);//Centered derivative

	  Linop_diffusion[k] = ( (1.+ D*dt*fcos* (1.-2.*zeta) + D*D * dt*dt 
				 *pow(fcos,2) *(-1+zeta)* zeta) )/( pow(1.- D *dt*fcos* zeta,2));
	}


	//initial condition for rho and theta
	initFromLastFiles( Ncopy,  Nx,  rho,  theta, rho0, upward);
   
  

   
   /*********************************************/
  /****************** STRING METHOD ********************/
  /***  Start with string method to obtain a relevant candidate for the instanton 
  ** This allows us to check of the action computed by gMAM is indeed smaller than the action of the string heteroclinic orbit */
  
  c = 0;
  
  while(c < Niterations_string + 1){
      
      /******** FIRST STEP: Field evolution ********/
      
      //compute the non linear contribution;
      FNL( NL_rho, rho, theta , Ncopy, Nx, dx, k0, k1, k2, k3);
      
    //Fourier transform	
    //real to complex
      for(i=0;i<Ncopy;i++){
	  	fftw_execute_dft_r2c(rho_r2c, rho[i], rhoFourier[i]);
	  	fftw_execute_dft_r2c(rho_r2c, NL_rho[i], NL_rhoFourier[i]);
      }
    
    
      // Evolution : 
      for(i=0;i<Ncopy;i++){
	  	for(k=0;k<Nx;k++){ //used mode
	      
	      rhoFourier[i][k] = Linop_diffusion[k]*rhoFourier[i][k] 
		  + dt*NL_rhoFourier[i][k];   
	  	}
      }

      //Fourier transform :
      for(i=0;i< Ncopy;++i){ 
	  fftw_execute_dft_c2r(rho_c2r, rhoFourier[i], rho[i]);
      }
     
      //Normalisation, on the IFFT: 
      for(i=0;i< Ncopy ;i++){
	  	for(j=0;j<Nx;j++){ 
	      	rho[i][j] *= 1./(double)(Nx);
	  	}
      }
    
    
    
      /************* SECOND STEP: String normalization **********/
	  
	  
      //compute string 
      stringLength = computeString( Ncopy,  Nx,  dx,  s, rho);
    
      //ponderation between states
      for(i=1;i<Ncopy-1;i++){
	  	lastPoint = 1;          //Start at least from previous interval
	  for(k=lastPoint;k<Ncopy;k++){// Index for s, old string s[k] goes from s[1] to 1
	      if( s[k-1]<= normalizedString[i] && normalizedString[i]< s[k]){
		  q = (normalizedString[i]-s[k-1])/(s[k]-s[k-1]);
		  // Update rho
		  for(x=0;x<Nx;x++){
		      rhoBis[i][x] = (1-q)*rho[k-1][x] + q*rho[k][x];
		  }
				
	      }
	      lastPoint = k;
	  	}	
      }
	
      for(i=1;i<Ncopy-1;i++){
	  	for(x=0;x<Nx;x++){
	      rho[i][x] = rhoBis[i][x];
	  	}
      }
    
    if(c > plotStep){
    /************* THIRD STEP (after a little while): Compute noise history **********/
    
    /*If the noise is additive Gaussian, then the noise history is obtained directly without the need of solving an equation.
    Otherwise, we have to solve:
    dTheta_dtau = lambda * drho_dt - dH_dtheta 
    Be careful of the value of lambda for the normalized length. */
    

	//compute lambda: 
	 //1) dynamics,
	 for(i=0;i<Ncopy;i++){
	 	for(x=0;x<Nx;x++){
			determinFlow[i][x] = D/(dx*dx) * (rho[i][(x+1)%Nx] + rho[i][(x+Nx-1)%Nx]-2.*rho[i][x]) 
			+ NL_rho[i][x] ;
		}
	}
	 // 2) norm
	for(i=0;i<Ncopy;i++){
		lambdaTime[i] = normL2(Nx, dx, determinFlow[i]) / stringLength;
	}
	
	
    //Compute theta, solve the equation d_\tau \theta = d\rho_ds - dH_d\theta, 
    //simple Euler scheme, gradient ascent
    dtau  = 0.05*dt; //artificial time step to solve theta
    error = 1.;
    for(i=0;i<Ncopy;i++){
     	if( i==0 || i==Ncopy-1){
     		for(x=0;x<Nx;x++){
    			theta[i][x]    = 0;
    			thetaBis[i][x] = 0;
    		}
     	}
    	else{
    		for(x=0;x<Nx;x++){
    			thetaBis[i][x] = theta[i][x] + dtau * ( lambdaTime[i]*(rho[(i+1)%Ncopy][x]-rho[i][x])/ds 
    			- ( 
    			D/(dx*dx) * (rho[i][(x+1)%Nx] + rho[i][(x+Nx-1)%Nx]-2.*rho[i][x])  
    			+ (k0 + k2*rho[i][x]*rho[i][x])*exp(theta[i][x])
    			- (k1*rho[i][x] + k3*rho[i][x]*rho[i][x]*rho[i][x])*exp(-theta[i][x])
    			-2*D/(dx*dx)* 
    				(0.5*(rho[i][x]+rho[i][(x+1)%Nx])*theta[i][(x+1)%Nx]
    				+0.5*(rho[i][x]+rho[i][(x-1+Nx)%Nx])*theta[i][(x-1+Nx)%Nx]
    				-0.5*(rho[i][(x+1)%Nx]+rho[i][(x-1+Nx)%Nx]+2*rho[i][x])*theta[i][x]	    					
    				)
    			) );
    		}
    	}
    }
    
    while(error > 1e-4){
    
    	error = 0;
    	for(i=0;i<Ncopy;i++){
     		if( i==0 || i==Ncopy-1){
     			for(x=0;x<Nx;x++){
    				theta[i][x]    = 0;
    				thetaBis[i][x] = 0;
    			}
     		}
    		else{
				for(x=0;x<Nx;x++){
					thetaBis[i][x] = theta[i][x] + dtau * ( lambdaTime[i]*(rho[(i+1)%Ncopy][x]-rho[i][x])/ds 
					- ( 
					D/(dx*dx) * (rho[i][(x+1)%Nx] + rho[i][(x+Nx-1)%Nx]-2.*rho[i][x])  
					+ (k0 + k2*rho[i][x]*rho[i][x])*exp(theta[i][x])
					- (k1*rho[i][x] + k3*rho[i][x]*rho[i][x]*rho[i][x])*exp(-theta[i][x])
					-2*D/(dx*dx)* 
						(0.5*(rho[i][x]+rho[i][(x+1)%Nx])*theta[i][(x+1)%Nx]
						+0.5*(rho[i][x]+rho[i][(x-1+Nx)%Nx])*theta[i][(x-1+Nx)%Nx]
						-0.5*(rho[i][(x+1)%Nx]+rho[i][(x-1+Nx)%Nx]+2*rho[i][x])*theta[i][x]	    					
						)
					) );
				}
    		}
    	}
    	
    	for(i=0;i<Ncopy;i++){
    		error += distanceL2( Nx,  dx,  theta[i],  thetaBis[i]);
    	}
		updateA1fromA2Real(theta, thetaBis,  Ncopy, Nx);
    }
    }
    
      //print result
      if((c%(plotStep)==0)){
		printRhoTXT( Ncopy,  Nx,  c, rho);
		printThetaTXT( Ncopy,  Nx,  c, theta);
		computeLagrangian( gMAMLagrangian,  rho,  theta,  Ncopy,  Nx,  ds, dx);
		printLagrangianString( Ncopy,  Nx,  c,  gMAMLagrangian);

		printf("file_%d\n", c);
      }
    
      c += 1;
    
  }//END while-time loop for STRING algorithm
  
  
  
  
  
  
  
  /**********************************************************************************/
  /*********** gMAM ALGORITHM (Hamiltonian), solution of Hyperbolic equation **************/
  /*********************************************************************************/
  
  /* alternate update of two hyperbolic equations.
  * We treat the diffusion implicitely w.r.t physical time.
  * The endpoints of the string become boundary conditions for the gMAM algorithm.
  * We split the diffusion of the fields. 
  * Define U = rho+theta, 
  			V = rho-theta
  * 1) Solve dU_dtau = dH_drho - dH_dtheta 
  	1a) semi-spectral semi-implicit solving
  	1b) renormalize distance (string on rho)
  	
  * 2) Solve dV_dtau = dH_drho + dH_dtheta
    2a) semi-spectral semi-implicit solving
  	2b) renormalize distance (string on rho)
  	
  	again until convergence...
  	
  */
  	
  	
	c = 0;
	dtau = 0.5*dt ; //smaller time step for gMAM GDA
  
    /**** Define vataibles specific to gMAM *****/
    double expT;
    double expTm;
    double r;
	r = dtau/ds;
    
    double normXdotSquare, scalarProduct;
    
    // Declare 2-D arrays (Ncopy,Nx)
    double** U;
    double** V;
    double** Vmean;
    double** Udiff;
    double** Umean; 
    double** Vdiff;
	double** dH_drho_noDiff;
	double** dH_dtheta_noDiff;
	double** dH_dtheta;
	double** dH_drho;
	double** reaction_V;
	double** reaction_U;
	
    fftw_complex** V_Fourier; 
  	fftw_complex** U_Fourier;
  	fftw_complex** tempArray;
  	fftw_complex** reaction_V_Fourier;
  	fftw_complex** reaction_U_Fourier;
  	
  	
  	// Declare 1-D arrays
  	double* rho1;
  	double* rho2;
  	double* Ak;
  	
  	fftw_complex* rho1k;
  	fftw_complex* rho2k;
  	fftw_complex* diag;
  	fftw_complex* lowerDiag; 
  	fftw_complex* upperDiag; 
  	fftw_complex* lower2ndDiag; 
  	fftw_complex* upper2ndDiag; 
  	fftw_complex* solX;
  	fftw_complex* rhs;
  	
  	
  	// Initialization of arrays
  	// 2-D arrays
  	U     = fftw_malloc(sizeof(double*) * Ncopy);
    V     = fftw_malloc(sizeof(double*) * Ncopy);
    Vmean = fftw_malloc(sizeof(double*) * Ncopy);
    Udiff = fftw_malloc(sizeof(double*) * Ncopy);
    Umean = fftw_malloc(sizeof(double*) * Ncopy); 
    Vdiff = fftw_malloc(sizeof(double*) * Ncopy);
	dH_drho_noDiff   = fftw_malloc(sizeof(double*) * Ncopy);
	dH_drho          = fftw_malloc(sizeof(double*) * Ncopy);
	dH_dtheta_noDiff = fftw_malloc(sizeof(double*) * Ncopy);
	dH_dtheta        = fftw_malloc(sizeof(double*) * Ncopy);
  	reaction_V       = fftw_malloc(sizeof(double*) * Ncopy);
  	reaction_U       = fftw_malloc(sizeof(double*) * Ncopy);
  	
  	U_Fourier    = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
    V_Fourier    = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
 	tempArray    = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
  	reaction_V_Fourier = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
  	reaction_U_Fourier = fftw_malloc(sizeof(fftw_complex*) * Ncopy); 
	
	for (i = 0; i < Ncopy; i++)
    {
        U[i]   = fftw_malloc(sizeof(double) * Nx);
        V[i]   = fftw_malloc(sizeof(double) * Nx);
        Vmean[i]  = fftw_malloc(sizeof(double) * Nx);
  	  	Udiff[i]  = fftw_malloc(sizeof(double) * Nx);
   		Umean[i]  = fftw_malloc(sizeof(double) * Nx);
    	Vdiff[i]  = fftw_malloc(sizeof(double) * Nx);
		dH_drho_noDiff[i]   = fftw_malloc(sizeof(double) * Nx);
		dH_drho[i]          = fftw_malloc(sizeof(double) * Nx);
		dH_dtheta_noDiff[i] = fftw_malloc(sizeof(double) * Nx);
		dH_dtheta[i]        = fftw_malloc(sizeof(double) * Nx);
  		reaction_V[i]       = fftw_malloc(sizeof(double) * Nx);
  		reaction_U[i]       = fftw_malloc(sizeof(double) * Nx);
        
        U_Fourier[i] = fftw_malloc(sizeof(fftw_complex) * Nx);
        V_Fourier[i] = fftw_malloc(sizeof(fftw_complex) * Nx);
        tempArray[i]          = fftw_malloc(sizeof(fftw_complex) * Nx); 
  		reaction_V_Fourier[i] = fftw_malloc(sizeof(fftw_complex) * Nx); 
  		reaction_U_Fourier[i] = fftw_malloc(sizeof(fftw_complex) * Nx); 

    }
    
    
	// 1-D arrays
  	rho1  = fftw_malloc(sizeof(double) * Nx);
  	rho2  = fftw_malloc(sizeof(double) * Nx);
  	 Ak   = fftw_malloc(sizeof(double) * Nx);
    rho1k = fftw_malloc(sizeof(fftw_complex) * Nx);
  	rho2k = fftw_malloc(sizeof(fftw_complex) * Nx);
  	
  	diag         = fftw_malloc(sizeof(fftw_complex) * Ncopy);
  	lowerDiag    = fftw_malloc(sizeof(fftw_complex) * Ncopy);
  	lower2ndDiag = fftw_malloc(sizeof(fftw_complex) * Ncopy);
    upperDiag    = fftw_malloc(sizeof(fftw_complex) * Ncopy);
    upper2ndDiag = fftw_malloc(sizeof(fftw_complex) * Ncopy);
  	solX         = fftw_malloc(sizeof(fftw_complex) * Ncopy);
  	rhs          = fftw_malloc(sizeof(fftw_complex) * Ncopy);

  
  	//To smoothen dynamics of stiff modes use array Ak, if needed
  	Ak[0] = 1;
  	for(k=0; k<Nx;k++){
  		//Ak[k] = 1./(1.+D/(dx*dx)*4*( sin(M_PI*k/((double) Nx))*sin(M_PI*k/((double) Nx)) ) );
  		Ak[k] = 1; 	
  	}
  	
  	//Initialization of boundary states. They are given by the end points of the string method
  	for(x=0;x<Nx;x++){
  	 	rho1[x] = rho[0][x];
  	 	rho2[x] = rho[Ncopy-1][x];
  	}
  	fftw_execute_dft_r2c(rho_r2c, rho1, rho1k );
  	fftw_execute_dft_r2c(rho_r2c, rho2, rho2k );
  	
  	//Initialization of U and V
  	sum(  rho,  theta, U,  Ncopy,  Nx) ; 	
  	diff( rho,  theta, V,  Ncopy,  Nx) ; 
  
  while(c < Niterations_gMAM+1){
  
  
    /*******************************************************************
  	/*********** STEP 1: solve  dU_dtau = dH_drho - dH_dtheta   ********/
  	
	
	
  	//Compute the model-dependent contribution (STEP 1)
      FNL( NL_rho, rho, theta , Ncopy, Nx, dx, k0, k1, k2, k3);

	

	for(i=0;i<Ncopy;i++){
	 	for(x=0;x<Nx;x++){
	 		expT  = exp(theta[i][x]); // convenient factors
	 		expTm = exp(-theta[i][x]);
	 		dH_drho[i][x] = D/(dx*dx)*(  exp(theta[i][(x+1)%Nx]-theta[i][x]) 
	 								+ exp(theta[i][(x-1+Nx)%Nx]-theta[i][x])
	 								- 2. )
	 								+ 2*k2*rho[i][x]*(expT-1)
	 								+ (k1 + 3*k3*rho[i][x]*rho[i][x])*(expTm-1);
	 								
			dH_dtheta[i][x] =  (k0 + k2*rho[i][x]*rho[i][x])*expT
					- (k1*rho[i][x] + k3*rho[i][x]*rho[i][x]*rho[i][x])*expTm
				+ D/(dx*dx)* 
					( -rho[i][x]*( exp(-theta[i][x]+theta[i][(x+1)%Nx]) 
									+ exp(-theta[i][x]+theta[i][(x-1+Nx)%Nx]) )
					+ rho[i][(x-1+Nx)%Nx]*exp(theta[i][x]-theta[i][(x-1+Nx)%Nx])
					+ rho[i][(x+1)%Nx]   *exp(theta[i][x]-theta[i][(x+1)%Nx]) ) ; 						
		}
	}


    for(i=0;i<Ncopy;i++){
	 	for(x=0;x<Nx;x++){
		  	reaction_U[i][x] = dH_drho[i][x] - dH_dtheta[i][x];
  		}	
  	}

  

  	//Compute Fourier transform of nonlinear terms (STEP 1)
  	//Real to complex
  	for(i=0;i<Ncopy;i++){
  		fftw_execute_dft_r2c(rho_r2c, reaction_U[i], reaction_U_Fourier[i] );
		fftw_execute_dft_r2c(rho_r2c, V[i], V_Fourier[i] );
		fftw_execute_dft_r2c(rho_r2c, U[i], U_Fourier[i] );
	}

 
 

	 // UPDATE lambda in Hamiltonian algorithm
  	
	lambdaTime[0]     = 0;
	lambdaTime[Ncopy] = 0;
	computeHamiltonian( gMAMHamiltonian, rho, theta, NL_rho, Ncopy, Nx, dx, D, k0, k1, k2, k3);
	
	for(i=1;i<Ncopy-1;i++){
		normXdotSquare = 0;
		scalarProduct  = 0;
		for(k=0;k<Nx;k++){
			normXdotSquare += dx * (rho[i+1][k]-rho[i][k])/ds * (rho[i+1][k]-rho[i][k])/ds;
		}
		for(k=0;k<Nx;k++){ 
			scalarProduct  += dx * (rho[i+1][k]-rho[i][k])/ds * dH_dtheta[i][k];
		}
	
		lambdaTime[i] = (scalarProduct + sqrt(returnMax(0,scalarProduct*scalarProduct-4*gMAMHamiltonian[i]*normXdotSquare)))
							/(2.0*normXdotSquare);
	}

  	

  	//Define tempArray from U with boundary conditions (STEP 1)
  	updateA1fromA2Complex(tempArray, U_Fourier, Ncopy, Nx);
  	
  	//Solve for a mode k (STEP 1) popagates from BC in i=Ncopy-1
  	for(k=0;k<Nx; k++){
  		//Boundary
  		tempArray[Ncopy-1][k]   = (-V_Fourier[Ncopy-1][k]+ 2*rho2k[k]) * Ak[k] * dtau + U_Fourier[Ncopy-1][k];
		reaction_U_Fourier[Ncopy-1][k] = 0 ;
  		diag[Ncopy-1]                  = 1 + dtau*Ak[k] ;
  		rhs[Ncopy-1]  = tempArray[Ncopy-1][k] + dtau * Ak[k] * reaction_U_Fourier[Ncopy-1][k];
  		// Second order hyperbolic discretization
  		for(i=0;i<Ncopy-2; i++){
  			diag[i]         = 1 + 3*r/2.* Ak[k]*lambdaTime[i];
  			upperDiag[i]    = -2*r*Ak[k]*lambdaTime[i];
  			upper2ndDiag[i] = r/2.*Ak[k]*lambdaTime[i];
  			rhs[i]       = tempArray[i][k] + dtau * Ak[k] * reaction_U_Fourier[i][k];
  		}
  		//Correct boundary terms, order 1
  		i = Ncopy-2;
	  	diag[i]         = 1 + r* Ak[k]*lambdaTime[i];
		upperDiag[i]    = -r*Ak[k]*lambdaTime[i];
		rhs[i]          = tempArray[i][k] + dtau * Ak[k] * reaction_U_Fourier[i][k];
  		
  		solveUpperDiagonalSystem2order( diag,  upperDiag, upper2ndDiag, solX, rhs, Ncopy);
  		
  		for(i=0;i<Ncopy; i++){
  			U_Fourier[i][k] = solX[i];
  		}
  		
  	}

  	
  	//Compute IFFT of U_Fourier (STEP 1) popagates from BC in i=Ncopy-1
  	  //Fourier transform :
      for(i=0;i< Ncopy;i++){ 
	  	fftw_execute_dft_c2r(rho_c2r, U_Fourier[i], U[i]);
      }
     
      //Normalisation, on the IFFT: 
      for(i=0;i< Ncopy ;i++){
	  	for(j=0;j<Nx;j++){ 
	      	U[i][j] *= 1./(double)(Nx);
	  	}
      }
  
  	linearCombi(0.5, U,  0.5, V, rho, Ncopy, Nx) ;
  	linearCombi(0.5, U, -0.5, V, theta, Ncopy, Nx) ;
  	
  	//Impose boundary conditions (STEP 1)
  	for(x=0;x<Nx;x++){
  		rho[0][x]       = rho1[x];
  		rho[Ncopy-1][x] = rho2[x];
  	  	theta[0][x]       = 0;
  		theta[Ncopy-1][x] = 0;
  	
  	}
  	
  	
  	/*********** Arclength reparametrization after update of U  (STEP 1) **************/
  	
  	  //Compute string 
      stringLength = computeString( Ncopy,  Nx,  dx,  s, rho);
    
      //ponderation between states
      for(i=1;i<Ncopy-1;i++){
	  	lastPoint = 1;          //Start at least from previous interval
	  for(k=lastPoint;k<Ncopy;k++){// Index for s, old string s[k] goes from s[1] to 1
	      if( s[k-1]<= normalizedString[i] && normalizedString[i]< s[k]){
		  q = (normalizedString[i]-s[k-1])/(s[k]-s[k-1]);
		  // Update rho
		  for(x=0;x<Nx;x++){
		      rhoBis[i][x]   = (1-q)*rho[k-1][x] + q*rho[k][x];
		      thetaBis[i][x] = (1-q)*theta[k-1][x] + q*theta[k][x];
		  }
				
	      }
	      lastPoint = k;
	  	}	
      }
	
	 updateA1fromA2RealExceptEndPoints(rho, rhoBis, Ncopy, Nx);
	 updateA1fromA2RealExceptEndPoints(theta, thetaBis, Ncopy, Nx);
	
	sum(  rho,  theta, U,  Ncopy,  Nx) ; 	
  	diff( rho,  theta, V,  Ncopy,  Nx) ; 
  

  

  	/*******************************************************************
  	/*********** STEP 2: solve  dV_dtau = dH_drho + dH_dtheta   ********/
  	
	
	
  	//Compute the model-dependent contribution (STEP 2) 
  	//Compute non diffusive part, useful when additive noise
      FNL( NL_rho, rho, theta , Ncopy, Nx, dx, k0, k1, k2, k3);
	
	for(i=0;i<Ncopy;i++){
	 	for(x=0;x<Nx;x++){
	 		expT  = exp(theta[i][x]);
	 		expTm = exp(-theta[i][x]);
	 		
	 		dH_drho[i][x] = 
	 					D/(dx*dx)*(  exp(theta[i][(x+1)%Nx]-theta[i][x]) 
	 								+ exp(theta[i][(x-1+Nx)%Nx]-theta[i][x])
	 								- 2. )
	 								+ 2*k2*rho[i][x]*(expT-1)
	 								+ (k1 + 3*k3*rho[i][x]*rho[i][x])*(expTm-1);
	 								
			dH_dtheta[i][x] =  (k0 + k2*rho[i][x]*rho[i][x])*expT
					- (k1*rho[i][x] + k3*rho[i][x]*rho[i][x]*rho[i][x])*expTm
				+ D/(dx*dx)* 
					( -rho[i][x]*( exp(-theta[i][x]+theta[i][(x+1)%Nx]) 
									+ exp(-theta[i][x]+theta[i][(x-1+Nx)%Nx]) )
					+ rho[i][(x-1+Nx)%Nx]*exp(theta[i][x]-theta[i][(x-1+Nx)%Nx])
					+ rho[i][(x+1)%Nx]   *exp(theta[i][x]-theta[i][(x+1)%Nx]) ) ; 						
		}
	}
  
  
    for(i=0;i<Ncopy;i++){
	 	for(x=0;x<Nx;x++){
		  	reaction_V[i][x] = dH_drho[i][x] + dH_dtheta[i][x];
  		}	
  	}
  
  
  	//Compute Fourier transform of nonlinear terms  (STEP 2)
  	//Real to complex
  	for(i=0;i<Ncopy;i++){
  		fftw_execute_dft_r2c(rho_r2c, reaction_V[i], reaction_V_Fourier[i] );
		fftw_execute_dft_r2c(rho_r2c, V[i], V_Fourier[i] );
		fftw_execute_dft_r2c(rho_r2c, U[i], U_Fourier[i] );
	}
  
  
	 // UPDATE lambda in Hamiltonian algorithm (STEP 2)
	lambdaTime[0]     = 0;
	lambdaTime[Ncopy] = 0;
	computeHamiltonian( gMAMHamiltonian, rho, theta, NL_rho, Ncopy, Nx, dx, D, k0, k1, k2, k3);
	
	for(i=1;i<Ncopy-1;i++){
		normXdotSquare = 0;
		scalarProduct  = 0;
		for(k=0;k<Nx;k++){
			normXdotSquare += dx * (rho[i+1][k]-rho[i][k])/ds * (rho[i+1][k]-rho[i][k])/ds;
		}
		for(k=0;k<Nx;k++){ 
			scalarProduct  += dx * (rho[i+1][k]-rho[i][k])/ds * dH_dtheta[i][k];
		}
		lambdaTime[i] = (scalarProduct + sqrt(returnMax(0,scalarProduct*scalarProduct-4*gMAMHamiltonian[i]*normXdotSquare)))
							/(2.0*normXdotSquare);
	}
  	
  	
  	
  	//Define tempArray with boundary conditions (STEP 2)
  	updateA1fromA2Complex(tempArray, V_Fourier, Ncopy, Nx);
  	
  	//Solve for a mode k (STEP 2) popagates from BC in i=0
  	for(k=0;k<Nx; k++){
  		//Boundary (STEP 2)
  		tempArray[0][k]          = (-U_Fourier[0][k]+ 2*rho1k[k]) * Ak[k] * dtau + V_Fourier[0][k];
		reaction_V_Fourier[0][k] = 0 ;
  		diag[0]                  = 1 + dtau*Ak[k] ;
  		rhs[0]       = tempArray[0][k] + dtau * Ak[k] * reaction_V_Fourier[0][k];
  		for(i=2;i<Ncopy; i++){
  			diag[i]      = 1 + 3/2.*r* Ak[k]*lambdaTime[i];
  			lowerDiag[i] = -2*r*Ak[k]*lambdaTime[i];
  			lower2ndDiag[i] = r/2.*Ak[k]*lambdaTime[i];
  			rhs[i]       = tempArray[i][k] + dtau * Ak[k] * reaction_V_Fourier[i][k];
  		}
  		// Correct boundary term
  		i = 1;
	  	diag[i]      = 1 + r* Ak[k]*lambdaTime[i];
		lowerDiag[i] = -r*Ak[k]*lambdaTime[i];
		
		rhs[i]       = tempArray[i][k] + dtau * Ak[k] * reaction_V_Fourier[i][k];
  		
  		solveLowerDiagonalSystem2order( diag,  lowerDiag, lower2ndDiag, solX, rhs, Ncopy);
  		
  		for(i=0;i<Ncopy; i++){
  			V_Fourier[i][k] = solX[i];
  		}
  		
  	}
  	
  	
  	//Compute IFFT of V_Fourier (STEP 2)
  	  //Fourier transform :
      for(i=0;i< Ncopy;i++){ 
	  	fftw_execute_dft_c2r(rho_c2r, V_Fourier[i], V[i]);
      }
     
      //Normalisation, on the IFFT (STEP 2)
      for(i=0;i< Ncopy ;i++){
	  	for(j=0;j<Nx;j++){ 
	      	V[i][j] *= 1./(double)(Nx);
	  	}
      }
  
  	linearCombi(0.5, U,  0.5, V, rho, Ncopy, Nx) ;
  	linearCombi(0.5, U, -0.5, V, theta, Ncopy, Nx) ;
  	
  	//Impose boundary conditions (STEP 2)
  	for(x=0;x<Nx;x++){
  		rho[0][x]       = rho1[x];
  		rho[Ncopy-1][x] = rho2[x];
  	  	theta[0][x]       = 0;
  		theta[Ncopy-1][x] = 0;
  	
  	}
  	
  	
  	
  	/******************** Arclength reparametrization after update of V (STEP 2) **************/
  	  //Compute string 
      stringLength = computeString( Ncopy,  Nx,  dx,  s, rho);
    
      //ponderation between states 
      for(i=1;i<Ncopy-1;i++){
	  	lastPoint = 1;          //Start at least from previous interval
	  for(k=lastPoint;k<Ncopy;k++){// Index for s, old string s[k] goes from s[1] to 1
	      if( s[k-1]<= normalizedString[i] && normalizedString[i]< s[k]){
		  q = (normalizedString[i]-s[k-1])/(s[k]-s[k-1]);
		  // Update rho
		  for(x=0;x<Nx;x++){
		      rhoBis[i][x]   = (1-q)*rho[k-1][x] + q*rho[k][x];
		      thetaBis[i][x] = (1-q)*theta[k-1][x] + q*theta[k][x];
		  }
				
	      }
	      lastPoint = k;
	  	}	
      }
	
	 updateA1fromA2RealExceptEndPoints(rho, rhoBis, Ncopy, Nx);
	 updateA1fromA2RealExceptEndPoints(theta, thetaBis, Ncopy, Nx);

	sum(  rho,  theta, U,  Ncopy,  Nx) ; 	
  	diff( rho,  theta, V,  Ncopy,  Nx) ; 
  	
  	
  	
  	
  	
  	  //Print rho and theta and Lagrangian
      if((c%plotStep)==0){
	  	printRhoTXTgMAM( Ncopy,  Nx,  c, rho);
	  	printThetaTXTgMAM( Ncopy,  Nx,  c, theta);
	  	computeHamiltonian( gMAMHamiltonian, rho, theta, NL_rho, Ncopy, Nx, dx, D, k0, k1, k2, k3);
	  	computeLagrangian( gMAMLagrangian,  rho,  theta,  Ncopy,  Nx,  ds, dx);
	  	printLagrangianGMAM( Ncopy,  Nx,  c,  gMAMLagrangian);
	  	printHamiltonianGMAM( Ncopy,  Nx,  c,  gMAMHamiltonian);
	  	printLambdaGMAM( Ncopy, Nx,  c,  lambdaTime);
	  	printf("file_%d\n", c);
      }
  
      c += 1;
    
  }//end while-time loop for gMAM ALGORITHM
  
  
  
  
  
  
  
	stop    = get_time();
	timeRun = stop-start;
	printf("Time of run %f\n",timeRun);
  
	/*Free pointers (String related) */
	free(rho);
	free(rhoBis);
	free(theta);
	free(thetaBis);
	free(determinFlow);

	free(rhoFourier);
	free(NL_rho);
	free(lambdaTime);
  
    /*Free pointers (gMAM related) */
	free(U);
	free(V);
	free(Vmean);
	free( Udiff);
	free( Umean); 
	free( Vdiff);
	free( dH_drho_noDiff);
	free( dH_drho);
	free( dH_dtheta_noDiff);
	free( dH_dtheta);
	free( reaction_V);
	free( reaction_U);

	free(V_Fourier); 
	free(U_Fourier);
	free(tempArray);
	free(reaction_V_Fourier);
	free(reaction_U_Fourier);

	free(rho1);
	free(rho2);
	free(Ak);

	free(rho1k);
	free(rho2k);
	free(diag);
	free(lowerDiag); 
	free(upperDiag); 
	free(lower2ndDiag);
	free(upper2ndDiag);
	free(solX);
	free(rhs);
	free(gMAMLagrangian);
	free(gMAMHamiltonian);
   
  
  return 0;
  
}







////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////




