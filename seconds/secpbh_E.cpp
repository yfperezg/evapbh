#include <iostream>              // Initūs canonicī c++
#include <string>                // Interpositio lineae
#include <fstream>               // Initūs exitūsque
#include <cstdlib>               // Bibliothecae canonicae ūtilitātum generalium
#include <stdio.h>               // Indecēs
#include <math.h>                // Mathēmatica prior
#include <vector>                // Vectōrēs 
#include <algorithm>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <iterator>
//
//    GNU Scientific Libraries
//
#include <gsl/gsl_sf_bessel.h>   // Bessel functions
#include <gsl/gsl_errno.h>       // Error reporting
#include <gsl/gsl_spline.h>      // Interpolation
#include <gsl/gsl_math.h>        // Mathematical basics
#include <gsl/gsl_integration.h> // Integration
#include <gsl/gsl_sf_log.h>      // Logarithm functions
#include <gsl/gsl_vector.h>      // Vectors
#include <gsl/gsl_matrix.h>      // Matrices
#include <gsl/gsl_rng.h>         // Random numbers
#include <gsl/gsl_randist.h>     // Random number Distributions
#include <gsl/gsl_sf_pow_int.h>  // Powers
#include <gsl/gsl_sf_gamma.h>    // Gamma function
#include <gsl/gsl_monte.h>       // Montecarlo Integration
#include <gsl/gsl_monte_plain.h> // Montecarlo - Plain Integration
#include <gsl/gsl_monte_miser.h> // Montecarlo - Miser Integration
#include <gsl/gsl_monte_vegas.h> // Montecarlo - Vegas Integration
#include <gsl/gsl_poly.h>        // Working with polynomials
#include <gsl/gsl_sf_erf.h>      // Error functions
#include <gsl/gsl_multimin.h>    // Minimization
#include <gsl/gsl_odeiv2.h>      // Ordinary Coupled Differential Equations
//
#include <armadillo>             // Bibliothecae Armadillo
//
//    Auxiliary files
//
using namespace std;
using namespace arma;
//
//
#define Nb_en_pp  250 // Numerus lacuum in energia particulae primariae
#define Nb_en_sp  500 // Numerus lacuum in energia particulae secondariae
//
//


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//                                                                                                                            //
//                           Programa para determinar o numero de eventos de  neutrinos da atmosfera                          //
//                                                                                                                            //
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//                                                     Programa Principal                                                     //
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

int main(void){

  cout << showpoint << std::setprecision(6);
  
  ifstream dataMM, data_primp_f, data_primp_s, data_primp_v;
  ofstream file, fileMM; 
  string entrada, saida, is; 
  string MMs, part_st_f, part_st_v, part_st_s;
  string gens, pros, st;// string for generator + process

  //-----------------------------------------------------------------------------//
  //                          Arquivos de entrada e saida                        //
  //-----------------------------------------------------------------------------//

  const int e = 11, mu = 13, tau = 15, nue = 12, numu = 14, nutau = 16;
  const int d = 1, u = 2, s = 3, c = 4, b = 5, t = 6;
  const int g = 21, gam = 22, Z = 23, W = 24, H = 25;

  int secn_par;

  cout << "\t" << endl;
  cout << "Optā quaeso particulam secundariam:" << endl;
  cout << "12 -> nu_e,  14 -> nu_mu,  16 -> nu_tau" << endl;
  cout << "11 -> e,  22 -> photon" << endl;
  cout << "\t" << endl;
  cout << "Se antiparticulae tabelaria computare velles, numeris negativis utere" << endl;
  
  cin >> secn_par;

  string prp_s = to_string(secn_par);

  ofstream file_sp; string sp_s_f;

  switch(abs(secn_par)){

  case u:     sp_s_f = "u";     break;
  case d:     sp_s_f = "d";     break;
  case s:     sp_s_f = "s";     break;
  case c:     sp_s_f = "c";     break;
  case b:     sp_s_f = "b";     break;
  case t:     sp_s_f = "t";     break;

  case e:     sp_s_f = "e";     break;
  case nue:   sp_s_f = "nue";   break;
  case mu:    sp_s_f = "mu";    break;
  case numu:  sp_s_f = "numu";  break;
  case tau:   sp_s_f = "tau";   break;
  case nutau: sp_s_f = "nutau"; break;

  case g:     sp_s_f = "gl";    break;
  case gam:   sp_s_f = "ph";   break;
  case Z:     sp_s_f = "Z";     break;
  case W:     sp_s_f = "W";     break;
  case H:     sp_s_f = "H";     break;
    
  }

  if(secn_par < 0) sp_s_f  = "a"+sp_s_f;

  double range_E[Nb_en_sp + 1];

  double en_sp_min = 0.; // Log10@Etru min --- energia minima particulae secondariae
  double en_sp_max = 6.; // Log10@Etru max --- --- energia minima particulae secondariae

  for(int i = 0; i <= Nb_en_sp; i++) range_E[i]  = pow(10., en_sp_min + (en_sp_max - en_sp_min)*i/Nb_en_sp); 

  double cthemin = 1.0;
  double cthemax = -1.0;
  double dcthe_sp = (cthemax - cthemin)/Nb_th_sp;


  //---------------------------------------------------------//
  //              Tabulāria cum numerīs eventōrum            //
  //---------------------------------------------------------//

  part_st_f = "../Asymm/Data/d2N_dEdOm_t=1000.0s_a*=0.999_th=0.0_test.txt"; //"../Asymm/Data/d3N_dEdOmdth_t=1000.0s_a*=0.999_th=0.0_t=0.000_test.txt"; //"../Asymm/Data/d3Nf_dEdOm_t=1000.0s_a*=0.0_bin.txt";

  data_primp_f.open(part_st_f.c_str(), ios::in);

  if (data_primp_f.fail()){
    cerr << "Tabulāria cum numerīs eventōrum particularium primarium invenire non possum!" << endl;
    exit(1);
  }
  
  part_st_v = "../Asymm/Data/d2N_dEdOm_t=1000.0s_a*=0.999_th=0.0_test.txt"; //"../Phot-Grav/Data/d2Nv_dEdOm_t=1000.0s_a*=0.999_bin.txt";

  data_primp_v.open(part_st_v.c_str(), ios::in);

  if (data_primp_v.fail()){
    cerr << "Tabulāria cum numerīs eventōrum invenire non possum!" << endl;
    exit(1);
  }

  
  part_st_s = "../Asymm/Data/d2N_dEdOm_t=1000.0s_a*=0.999_th=0.0_test.txt"; //"../Phot-Grav/Data/d2Ns_dEdOm_t=1000.0s_a*=0.999_bin.txt";

  data_primp_s.open(part_st_s.c_str(), ios::in);

  if (data_primp_s.fail()){
    cerr << "Tabulāria cum numerīs eventōrum invenire non possum!" << endl;
    exit(1);
  }

  // Ēventum neutrinōrum apud energiam veram angulumque zenith

  std::vector<double> E_pp(Nb_en_pp);
  arma::vec d2ndEdO_pp_s(Nb_en_pp), d2ndEdO_pp_f(Nb_en_pp), d2ndEdO_pp_v(Nb_en_pp);

  for (int i = 0; i < Nb_en_pp; i++) {data_primp_s >> E_pp[i]; data_primp_s >> d2ndEdO_pp_s(i);}
 
  data_primp_s.close();


  for (int i = 0; i < Nb_en_pp; i++) {data_primp_v >> E_pp[i]; data_primp_v >> d2ndEdO_pp_v(i);}
 
  data_primp_v.close();
  
  
  for (int i = 0; i < Nb_en_pp; i++) {data_primp_f >> E_pp[i]; data_primp_f >> d2ndEdO_pp_f(i);}

  data_primp_f.close();

  //=========================================//
  //            Matrices definimus           //
  //      ad usum Armadillo bibliothecae     //
  //=========================================//

  cout << d2ndEdO_pp_f << endl;
  
  // Quarks

  arma::mat MM_pp_u(Nb_en_sp, Nb_en_pp), MM_pp_d(Nb_en_sp, Nb_en_pp),
    MM_pp_s(Nb_en_sp, Nb_en_pp), MM_pp_c(Nb_en_sp, Nb_en_pp),
    MM_pp_t(Nb_en_sp, Nb_en_pp), MM_pp_b(Nb_en_sp, Nb_en_pp);

  // Leptons

  arma::mat MM_pp_mu(Nb_en_sp, Nb_en_pp), MM_pp_tau(Nb_en_sp, Nb_en_pp);

  // Gauge Bosons
  
  arma::mat MM_pp_W(Nb_en_sp, Nb_en_pp), MM_pp_Z(Nb_en_sp, Nb_en_pp), MM_pp_gl(Nb_en_sp, Nb_en_pp);

  // Higgs

  arma::mat MM_pp_H(Nb_en_sp, Nb_en_pp);
 
  MM_pp_u.load("./Data/u/MM_u_"+sp_s_f+"_E.bin"); MM_pp_d.load("./Data/d/MM_d_"+sp_s_f+"_E.bin");
  MM_pp_s.load("./Data/s/MM_s_"+sp_s_f+"_E.bin"); MM_pp_c.load("./Data/c/MM_c_"+sp_s_f+"_E.bin");
  MM_pp_t.load("./Data/t/MM_t_"+sp_s_f+"_E.bin"); MM_pp_b.load("./Data/b/MM_b_"+sp_s_f+"_E.bin");

  MM_pp_gl.load("./Data/gl/MM_gl_"+sp_s_f+"_E.bin");

  MM_pp_mu.load("./Data/mu/MM_mu_"+sp_s_f+"_E.bin"); MM_pp_tau.load("./Data/tau/MM_tau_"+sp_s_f+"_E.bin");

  MM_pp_W.load("./Data/W/MM_W_"+sp_s_f+"_E.bin"); MM_pp_Z.load("./Data/Z/MM_Z_"+sp_s_f+"_E.bin"); MM_pp_H.load("./Data/H/MM_H_"+sp_s_f+"_E.bin");

  arma::mat MM_pp = MM_pp_u + MM_pp_d + MM_pp_s + MM_pp_c + MM_pp_t + MM_pp_b + MM_pp_mu + MM_pp_tau + MM_pp_W + MM_pp_Z + MM_pp_gl + MM_pp_H;

  MM_pp.save("./Data/MM_pp_numu_E.dat", raw_ascii);
 
  //.............................//
  //      Tabulariæ Exitūs      //
  //.............................//

  saida = "../Asymm/Data/d2N"+sp_s_f+"_dEdth_t=1000.0s_a*=0.999_secs_E.txt";
  
  file.open(saida.c_str(), ios::out);
  file.precision(10);

  file << scientific << showpoint; 
  
  //----------------------------------------------------------------------------//
  //                Computation of the Number of Events in the bin              //
  //----------------------------------------------------------------------------//

  arma::vec d2ndEdO_u_sp(Nb_en_sp), d2ndEdO_d_sp(Nb_en_sp),
            d2ndEdO_s_sp(Nb_en_sp), d2ndEdO_c_sp(Nb_en_sp),
            d2ndEdO_t_sp(Nb_en_sp), d2ndEdO_b_sp(Nb_en_sp);
  
  d2ndEdO_u_sp = MM_pp_u * d2ndEdO_pp_f; d2ndEdO_d_sp = MM_pp_d * d2ndEdO_pp_f;
  d2ndEdO_s_sp = MM_pp_s * d2ndEdO_pp_f; d2ndEdO_c_sp = MM_pp_c * d2ndEdO_pp_f;
  d2ndEdO_t_sp = MM_pp_t * d2ndEdO_pp_f; d2ndEdO_b_sp = MM_pp_b * d2ndEdO_pp_f;

  arma::vec d2ndEdO_mu_sp(Nb_en_sp), d2ndEdO_tau_sp(Nb_en_sp);

  d2ndEdO_mu_sp = MM_pp_mu * d2ndEdO_pp_f; d2ndEdO_tau_sp = 2. * MM_pp_tau * d2ndEdO_pp_f;
  
  arma::vec d2ndEdO_W_sp(Nb_en_sp), d2ndEdO_Z_sp(Nb_en_sp), d2ndEdO_gl_sp(Nb_en_sp), d2ndEdO_H_sp(Nb_en_sp);

  d2ndEdO_W_sp  = MM_pp_W * d2ndEdO_pp_v;  d2ndEdO_Z_sp = MM_pp_Z * d2ndEdO_pp_v;
  d2ndEdO_gl_sp = MM_pp_gl * d2ndEdO_pp_v;
  d2ndEdO_H_sp  = MM_pp_H * d2ndEdO_pp_v;
  
  arma::vec d2ndEdO_sp(Nb_en_sp);

  d2ndEdO_sp = (2. * 3 * (d2ndEdO_u_sp + d2ndEdO_u_sp + d2ndEdO_d_sp + d2ndEdO_s_sp + d2ndEdO_c_sp + d2ndEdO_t_sp + d2ndEdO_b_sp) + 
		2. * (d2ndEdO_mu_sp + d2ndEdO_tau_sp) +
		0.*2. * d2ndEdO_W_sp +  d2ndEdO_Z_sp + 8. * d2ndEdO_gl_sp +  2. * 4. * d2ndEdO_H_sp);

  cout << d2ndEdO_sp << endl;


  //+++++++++++++++++++++++++++++++
  //   Reconstructed energy loop
  //+++++++++++++++++++++++++++++++
  
  for (int i = 0; i < Nb_en_sp; i++){
      
    cout << range_E[i] << '\t' <<  d2ndEdO_sp(i) << endl;
    
    
    file << range_E[i] << '\t' <<  d2ndEdO_sp(i) << endl;
	    
  }
   
  //-----------------------------------------------------------------------------//
  //                            Fechamos as librarias                            //
  //-----------------------------------------------------------------------------//

  
  file.close();

  cout << endl;
  
  return 0;
  
}