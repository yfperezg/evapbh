//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//                                                                                                                                   //
//                                         Particulae Secondariae e foraminibus atris emissae                                        //
//                                                    Matricis Migrationis Computatio                                                //
//                                                                                                                                   //
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//
#include <iostream>              // Initūs canonicī c++
#include <string>                // Interpositio lineae
#include <fstream>               // Initūs exitūsque
#include <cstdlib>               // Bibliothecae canonicae ūtilitātum generalium
#include <stdio.h>               // Indecēs
#include <math.h>                // Mathēmatica prior
#include <vector>                // Vectōrēs 
#include <algorithm>
#include <array>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <complex>
#include <cmath>
#include <chrono>
//
//    GNU Bibliothecae scientificae
//
#include <gsl/gsl_sf_bessel.h>   // Fūnctiōnēs Besselis
#include <gsl/gsl_errno.h>       // Audītiō erratōrum
#include <gsl/gsl_spline.h>      // Interpolātiō
#include <gsl/gsl_math.h>        // Mathēmatica prior
#include <gsl/gsl_integration.h> // Integrātiōnēs
#include <gsl/gsl_sf_log.h>      // Logarithmī
#include <gsl/gsl_vector.h>      // Vectōrēs
#include <gsl/gsl_matrix.h>      // Mātrīcēs
#include <gsl/gsl_rng.h>         // Numerī āleātōriī
#include <gsl/gsl_randist.h>     // Distribūtiōnēs numerōrum āleātōriōrum
#include <gsl/gsl_sf_pow_int.h>  // Potentiae
#include <gsl/gsl_sf_gamma.h>    // Fūnctio gamma
#include <gsl/gsl_monte.h>       // Montecarlo Integrātiō
#include <gsl/gsl_monte_plain.h> // Montecarlo - Plain Integrātiō
#include <gsl/gsl_monte_miser.h> // Montecarlo - Miser Integrātiō
#include <gsl/gsl_monte_vegas.h> // Montecarlo - Vegas Integrātiō
#include <gsl/gsl_poly.h>        // Polynomia
#include <gsl/gsl_sf_erf.h>      // Fūnctio erratōrum
#include <gsl/gsl_multimin.h>    // Minima invenire
#include <gsl/gsl_histogram2d.h> // 2D Histograma
//
#include <armadillo>             // Bibliothecae Armadillo
//
//
#define Nb_en_pp  50  // Numerus lacuum in energia particulae primariae
#define Nb_th_pp  50  // Numerus lacuum in cos(angulo) particulae primariae

#define Nb_del    100 // Numerus lacuum in angulo relativo

#define Nb_en_sp  50  // Numerus lacuum in energia particulae secondariae
#define Nb_th_sp  50  // Numerus lacuum in cos(angulo) particulae secondariae
//
//

using namespace std;
using namespace arma;

//-----------------------------------------------------------------------------------//
//                           Programma Praecĭpŭum
//-----------------------------------------------------------------------------------//

 int main() {

   //std::cout<< scientific << showpoint; 

  const int e = 11, mu = 13, tau = 15, nue = 12, numu = 14, nutau = 16;
  const int d = 1, u = 2, s = 3, c = 4, b = 5, t = 6;
  const int g = 21, gam = 22, Z = 23, W = 24, H = 25;

  int prim_par;

  cout << "\t" << endl;
  cout << "Optā quaeso particulam primariam:" << endl;
  cout << "1  -> d,   2 -> u,     3 -> s,  4 -> c,  5 -> b,  6 -> t" << endl;
  cout << "11 -> e,  13 -> mu,   15 -> tau" << endl;
  cout << "21 -> gl, 22 -> phot, 23 -> Z, 24 -> W, 25 -> H" << endl;
  cout << "\t" << endl;
  
  cin >> prim_par;

  //prim_par = 1;

  string prp_s = to_string(prim_par);

  ofstream file_pp, file_app; string pp_s_f;

  file_pp.open("test.txt", ios::out); file_pp.precision(10);

  switch(prim_par){

  case u:     pp_s_f = "u";     break;
  case d:     pp_s_f = "d";     break;
  case s:     pp_s_f = "s";     break;
  case c:     pp_s_f = "c";     break;
  case b:     pp_s_f = "b";     break;
  case t:     pp_s_f = "t";     break;

  case e:     pp_s_f = "e";     break;
  case nue:   pp_s_f = "nue";   break;
  case mu:    pp_s_f = "mu";    break;
  case numu:  pp_s_f = "numu";  break;
  case tau:   pp_s_f = "tau";   break;
  case nutau: pp_s_f = "nutau"; break;

  case g:     pp_s_f = "gl";    break;
  case gam:   pp_s_f = "gam";   break;
  case Z:     pp_s_f = "Z";     break;
  case W:     pp_s_f = "W";     break;
  case H:     pp_s_f = "H";     break;
    
  }

  cout << "Particula primaria = " << pp_s_f << endl;

  //---------------------------------------------------------//
  //              Migration Matrix - Input files             //
  //--------------------------------------------------------//

  auto begin = std::chrono::system_clock::now();
  //std::time_t begin_time = std::chrono::system_clock::to_time_t(begin);

  string MM_pp_s;     // Strings for Migration matrices input - CC1l0p
  
  MM_pp_s = "./Data/"+pp_s_f+"/MM-"+pp_s_f+".txt";

  ifstream data_pp;
    
  data_pp.open(MM_pp_s.c_str(), ios::in); data_pp.precision(10);

  if (data_pp.fail()){
    cerr << "Mātrix Migrātionis invenire nōn possum!" << endl;
    exit(1);
  }
  
  const int NMM = Nb_en_sp*(Nb_en_pp+1)*Nb_del;
  std::vector<std::vector<double>> MM_pp(NMM, std::vector<double>(9));
  
  // Particularum histogramae in energia anguloque

  double range_E[Nb_en_sp + 1];
  double range_del[Nb_del + 1];
  
  double en_pp_min = 2.; // Log10@Etru min --- energia minima particulae primariae
  double en_pp_max = 6.; // Log10@Etru max --- energia maxima particulae primariae

  double en_sp_min = 0.;  // Log10@Etru min --- energia minima particulae secondariae
  double en_sp_max = 6.;  // Log10@Etru min --- energia minima particulae secondariae

  double delmin = -1.,  delmax = 1.; // Cos anguli relativi

  for(int i = 0; i <= Nb_en_sp; i++) range_E[i]  = pow(10., en_sp_min + (en_sp_max - en_sp_min)*i/Nb_en_sp); 
  for(int i = 0; i <= Nb_del; i++)  range_del[i] = delmin + (delmax - delmin)*i/Nb_del;
  
  for(int k = 0; k < NMM; k++) 
      for(int l = 0; l < 9; l++)
	data_pp >> MM_pp[k][l];

  data_pp.close();

  auto end = std::chrono::system_clock::now();
      
  std::chrono::duration<double> elapsed_seconds = end-begin;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  auto ed_time = localtime(&end_time);
  auto endtm  = std::put_time(ed_time, "%H:%M:%S ante diem %d-%m-%Y");

  double elap_h, elap_m, elap_s;

  elap_h = floor(elapsed_seconds.count()/3600.);
  elap_m = floor(fmod(elapsed_seconds.count(), 3600.)/60.);
  elap_s = fmod(elapsed_seconds.count(), 60.);

  cout << " " << endl;
  cout << "Matricēs e Pythia lectae sunt. Tempus intercessum: "
       << elap_h << " h "<< elap_m << " min " << elap_s << " s. Ab hōrā "
       << endtm << " perfecit" << endl << endl;

  cout << '\t' << endl;

  //====================================================================================================================================================//
  //                                            Matrices migrationis definimus ut Armadillo bibliotheca eis utatur                                      //
  //====================================================================================================================================================//
  
  // Particulae secondariae e particula primaria

  arma::sp_mat MM_pp_ph(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp),
    MM_pp_ep(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp), MM_pp_em(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp),
    MM_pp_nue(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp), MM_pp_anue(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp),
    MM_pp_numu(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp), MM_pp_anumu(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp),
    MM_pp_nutau(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp), MM_pp_anutau(Nb_en_sp*Nb_th_sp, Nb_en_pp*Nb_th_pp);
  

  //------------------------------------------------------------------//
  //                Lacus in energia anguloque facemus               //
  //------------------------------------------------------------------//
  
  // Numerum minimum maximumque angulo 
  double cthemin = -1.0;
  double cthemax =  1.0;

  double dcthe_pp = (cthemax - cthemin)/Nb_th_pp; 
  double dcthe_sp = (cthemax - cthemin)/Nb_th_sp;

  // Minimum and maximum values of the auxiliar azimuthal angle phi
  double cphi;
  const int Nphi = 100;
  double cphimin = 0.;
  double cphimax = 2.*M_PI;

  double dcphi = (cphimax - cphimin)/Nphi;

  //----------------------------------------------------------------------------//
  //                Computation of the Migration matrix for the                 //
  //                          reconstructed nadir angle                         //
  //----------------------------------------------------------------------------//

  auto start_MM = std::chrono::system_clock::now();

  double Enu_pp_i, Enu_pp_f, Enu_sp_i, Enu_sp_f, cthe_pp_i, cthe_sp_i, cthe_pp_f, cthe_sp_f;

  double Evsrt_pp_ph, Evsrt_pp_em, Evsrt_pp_ep,
    Evsrt_pp_nue, Evsrt_pp_numu, Evsrt_pp_nutau,
    Evsrt_pp_anue, Evsrt_pp_anumu, Evsrt_pp_anutau;


  //++++++++++++++++++++++++++++++++++++++++++++
  //   Lăquĕus energiae particula primariae
  //++++++++++++++++++++++++++++++++++++++++++++
  
  for (int epp = 0; epp < Nb_en_pp; epp++){

  //int epp = 25;

    Enu_pp_i = pow(10., en_pp_min + (en_pp_max - en_pp_min)*epp/Nb_en_pp);
    Enu_pp_f = pow(10., en_pp_min + (en_pp_max - en_pp_min)*(epp+1)/Nb_en_pp);

    cout << epp << '\t' << Enu_pp_i << '\t' << Enu_pp_f << endl;
        
    //++++++++++++++++++++++++++++++++++++++++++++++++
    //      Lăquĕus energiae particula secondariae
    //++++++++++++++++++++++++++++++++++++++++++++++++
    
    for (int k = 0; k < Nb_en_sp; k++){

      //int k = 0;
      
      Enu_sp_i = range_E[k];
      Enu_sp_f = range_E[k+1];

      //cout << l << '\t' << Enu_pp_i << endl;
      
      //+++++++++++++++++++++++++++++++++++++++++++++++++
      //       Lăquĕus cos(anguli) particula primariae
      //+++++++++++++++++++++++++++++++++++++++++++++++++
      
      for (int j = 0; j < Nb_th_pp; j++){

	//int j = 27; 
	
	cthe_pp_i = cthemin + j*dcthe_pp;
	cthe_pp_f = cthemin + (j + 1)*dcthe_pp;

	//cout <<  cthe_pp_i - cthe_pp_f << endl;

	//++++++++++++++++++++++++++++++++++++++++++++++++
	//   Lăquĕus cos(anguli) particula secondariae
	//++++++++++++++++++++++++++++++++++++++++++++++++
	
	for(int m = 0; m < Nb_th_sp; m++){

	  //int m = 1;
	  
	  cthe_sp_i = cthemin + m * dcthe_sp;
	  cthe_sp_f = cthemin + (m + 1) * dcthe_sp;
	 
	  Evsrt_pp_ph = 0.; Evsrt_pp_em = 0.; Evsrt_pp_ep = 0.;
	  Evsrt_pp_nue = 0.; Evsrt_pp_numu = 0.; Evsrt_pp_nutau = 0.;
	  Evsrt_pp_anue = 0.; Evsrt_pp_anumu = 0.; Evsrt_pp_anutau = 0.;
	  
	  double the_pp = 0.5*(cthe_pp_f + cthe_pp_i);
	 
	  for (int l = 0; l < Nb_del; l++){ // l -> relative angle index

	    //int l = 0;
	      
	      for (int p = 0; p < Nphi; p++){// p -> azimuthal angle index
		
		cphi = cphimin + p*dcphi;		

		double cthe_sp = (the_pp * range_del[l] - sqrt(1. - the_pp*the_pp) * sqrt(1. - range_del[l]*range_del[l]) * cos(cphi));
		
		//cout << the_pp << '\t' << range_del[l] <<'\t' << cphi << '\t' << cthe_sp << '\t' << MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][3] <<  endl;

		
		if (cthe_sp_i <= cthe_sp && cthe_sp < cthe_sp_f) {

		  //cout << the_pp << '\t' << range_del[l] <<'\t' << cphi << '\t' << cthe_sp << '\t' << l << '\t' << MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][3] <<  endl;

		  //test_a[j][m] += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][3]/(Nphi+1);

		  //cout <<cthe_sp_f << '\t' <<  cthe_sp_i << "  here" <<endl;
		  
		  Evsrt_pp_ph     += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][0]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_ep     += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][1]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_em     += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][2]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_nue    += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][3]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_anue   += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][4]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_numu   += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][5]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_anumu  += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][6]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_nutau  += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][7]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  Evsrt_pp_anutau += MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][8]/(Enu_sp_f - Enu_sp_i)/abs(dcthe_pp)/(Nphi);
		  
		  }    
		
	      }	 
	      
	  }
	  
	  // Matrices definimus 
	  
	  MM_pp_ph(m*Nb_en_sp + k,     j*Nb_en_pp + epp) = Evsrt_pp_ph;
	  MM_pp_em(m*Nb_en_sp + k,     j*Nb_en_pp + epp) = Evsrt_pp_em;
	  MM_pp_ep(m*Nb_en_sp + k,     j*Nb_en_pp + epp) = Evsrt_pp_ep;
	  
	  MM_pp_nue(m*Nb_en_sp + k,    j*Nb_en_pp + epp) = Evsrt_pp_nue;
	  MM_pp_anue(m*Nb_en_sp + k,   j*Nb_en_pp + epp) = Evsrt_pp_anue;
	  
	  MM_pp_numu(m*Nb_en_sp + k,   j*Nb_en_pp + epp) = Evsrt_pp_numu;
	  MM_pp_anumu(m*Nb_en_sp + k,  j*Nb_en_pp + epp) = Evsrt_pp_anumu;
	  
	  MM_pp_nutau(m*Nb_en_sp + k,  j*Nb_en_pp + epp) = Evsrt_pp_nutau;
	  MM_pp_anutau(m*Nb_en_sp + k, j*Nb_en_pp + epp) = Evsrt_pp_anutau;
	  
	}
	
      }
      
    }
    
  }
	    
  int epp = 25; int k = 0;
  for (int j = 0; j < Nb_th_pp; j++)
    for(int m = 0; m < Nb_th_sp; m++)
      {
	
	file_pp << cthemin + (j + 0.5)*dcthe_pp << '\t' << cthemin + (m + 0.5) * dcthe_sp  << '\t'
		<< MM_pp_ph(m*Nb_en_sp + k, j*Nb_en_pp + epp)    << '\t'
		<< MM_pp_nue(m*Nb_en_sp + k, j*Nb_en_pp + epp)   << '\t'
		<< MM_pp_anue(m*Nb_en_sp + k, j*Nb_en_pp + epp)  << '\t'
		<< MM_pp_numu(m*Nb_en_sp + k, j*Nb_en_pp + epp)  << '\t'
		<< MM_pp_anumu(m*Nb_en_sp + k, j*Nb_en_pp + epp) << '\t'
		<< MM_pp_nutau(m*Nb_en_sp + k, j*Nb_en_pp + epp) << '\t'
		<< MM_pp_anutau(m*Nb_en_sp + k, j*Nb_en_pp + epp)
		<< endl;
      }

  MM_pp_ph.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_gam.bin");
  MM_pp_em.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_em.bin");
  MM_pp_ep.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_ep.bin");

  MM_pp_nue.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_nue.bin");
  MM_pp_anue.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_anue.bin");   
  MM_pp_numu.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_numu.bin");
  MM_pp_anumu.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_anumu.bin"); 
  MM_pp_nutau.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_nutau.bin");
  MM_pp_anutau.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_anutau.bin"); 
	    
	  
  auto end_MM = std::chrono::system_clock::now();
      
  std::chrono::duration<double> elapsed_seconds_MM = end_MM - start_MM;
  std::time_t end_time_MM = std::chrono::system_clock::to_time_t(end_MM);

  auto MM_time = localtime(&end_time_MM);
  auto MMtm  = std::put_time(MM_time, "%H:%M:%S ante diem %d-%m-%Y");

  double elap_h_MM, elap_m_MM, elap_s_MM;

  elap_h_MM = floor(elapsed_seconds_MM.count()/3600.);
  elap_m_MM = floor(fmod(elapsed_seconds_MM.count(), 3600.)/60.);
  elap_s_MM = fmod(elapsed_seconds_MM.count(), 60.);
  
  
  cout << endl << endl;
  cout << "Mātrices Mīgrātiōnis ē Pythia factae sunt." << endl << endl;
  cout << "Tempus intercessum: "
       << elap_h_MM << " h "<< elap_m_MM << " min " << elap_s_MM << " s." << endl << endl
       << "Programma ab hōrā " << MMtm << " perfecit." << endl;
  cout << endl;

  file_pp.close();
  
  return 0;
  
}
