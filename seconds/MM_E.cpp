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
#define Nb_en_pp  250  // Numerus lacuum in energia particulae primariae
#define Nb_en_sp  500 // Numerus lacuum in energia particulae secondariae
//
//

using namespace std;
using namespace arma;

//-----------------------------------------------------------------------------------//
//                           Programma Praecĭpŭum
//-----------------------------------------------------------------------------------//

 int main() {

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

  string prp_s = to_string(prim_par);

  ofstream file_pp, file_app; string pp_s_f;

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

  //---------------------------------------------------------//
  //              Migration Matrix - Input files             //
  //--------------------------------------------------------//

  auto begin = std::chrono::system_clock::now();
  std::time_t begin_time = std::chrono::system_clock::to_time_t(begin);

  string MM_pp_s;     // Strings for Migration matrices input - CC1l0p
  
  MM_pp_s = "./Data/"+pp_s_f+"/MM-"+pp_s_f+"_E.txt";

  ifstream data_pp;
    
  data_pp.open(MM_pp_s.c_str(), ios::in); data_pp.precision(10);

  if (data_pp.fail()){
    cerr << "Mātrix Migrātionis invenire nōn possum!" << endl;
    exit(1);
  }

  file_pp.open("./Data/"+pp_s_f+"/MM_example.dat", ios::out); file_pp.precision(10);

  
  const int NMM = Nb_en_sp*(Nb_en_pp+1);
  std::vector<std::vector<double>> MM_pp(NMM, std::vector<double>(9));
  
  // Particularum histogramae in energia anguloque

  double range_E[Nb_en_sp + 1];
  
  double en_pp_min = log10(5.); // Log10@Etru min --- energia minima particulae primariae
  double en_pp_max = 5.; // Log10@Etru max --- energia maxima particulae primariae

  double en_sp_min = -6.; // Log10@Etru min --- energia minima particulae secondariae
  double en_sp_max = 5.; // Log10@Etru max --- --- energia minima particulae secondariae

  for(int i = 0; i <= Nb_en_sp; i++) range_E[i]  = pow(10., en_sp_min + (en_sp_max - en_sp_min)*i/(Nb_en_sp-1));
  
  for(int k = 0; k < NMM; k++) 
      for(int l = 0; l < 9; l++)
	data_pp >> MM_pp[k][l];

  data_pp.close();

  auto end = std::chrono::system_clock::now();
      
  std::chrono::duration<double> elapsed_seconds = end-begin;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  auto ed_time = localtime(&end_time);
  auto endtm  = std::put_time(ed_time, "%H:%M:%S ante diem %d-%m-%Y");

  cout << " " << endl;
  cout << "Matricēs e Pythia lectae sunt. Tempus intercessum: " << elapsed_seconds.count() << " s = "
       << elapsed_seconds.count()/60. << " min = " << elapsed_seconds.count()/3600. << " h. Ab hōrā "
       << endtm << " perfecit" << endl << endl;

  cout << '\t' << endl;

  //====================================================================================================================================================//
  //                                              Matrices migrationis definimus ut Armadillo bibliotheca utatur                                        //
  //====================================================================================================================================================//
  
  // Particulae secondariae e particula primaria

  arma::mat MM_pp_ph(Nb_en_sp, Nb_en_pp),
    MM_pp_ep(Nb_en_sp, Nb_en_pp), MM_pp_em(Nb_en_sp, Nb_en_pp),
    MM_pp_nue(Nb_en_sp, Nb_en_pp), MM_pp_anue(Nb_en_sp, Nb_en_pp),
    MM_pp_numu(Nb_en_sp, Nb_en_pp), MM_pp_anumu(Nb_en_sp, Nb_en_pp),
    MM_pp_nutau(Nb_en_sp, Nb_en_pp), MM_pp_anutau(Nb_en_sp, Nb_en_pp);
  
  //----------------------------------------------------------------------------//
  //                Computation of the Migration matrix for the                 //
  //                          reconstructed nadir angle                         //
  //----------------------------------------------------------------------------//

  auto start_MM = std::chrono::system_clock::now();

  double Enu_pp_i, Enu_sp_i, Enu_sp_f, cthe_pp_i, cthe_sp_i, cthe_pp_f, cthe_sp_f;

  double Evsrt_pp_ph, Evsrt_pp_em, Evsrt_pp_ep,
    Evsrt_pp_nue, Evsrt_pp_numu, Evsrt_pp_nutau,
    Evsrt_pp_anue, Evsrt_pp_anumu, Evsrt_pp_anutau;


  //++++++++++++++++++++++++++++++++++++++++++++
  //   Lăquĕus energiae particula primariae
  //++++++++++++++++++++++++++++++++++++++++++++
  
  for (int epp = 0; epp < Nb_en_pp; epp++){

    Enu_sp_i = pow(10., en_pp_min + (en_pp_max - en_pp_min)*epp/(Nb_en_pp-1));
    Enu_sp_f = pow(10., en_pp_min + (en_pp_max - en_pp_min)*(epp+1)/(Nb_en_pp-1));

    cout << epp << '\t' << Enu_sp_i << endl;
        
    //++++++++++++++++++++++++++++++++++++++++++++++++
    //      Lăquĕus energiae particula secondariae
    //++++++++++++++++++++++++++++++++++++++++++++++++
    
    for (int k = 0; k < Nb_en_sp; k++){

      //int l = 7;
      
      Enu_pp_i = range_E[k];
		  
      Evsrt_pp_ph     = MM_pp[epp*Nb_en_sp + k][0];
      Evsrt_pp_ep     = MM_pp[epp*Nb_en_sp + k][1];
      Evsrt_pp_em     = MM_pp[epp*Nb_en_sp + k][2];
      Evsrt_pp_nue    = MM_pp[epp*Nb_en_sp + k][3];
      Evsrt_pp_anue   = MM_pp[epp*Nb_en_sp + k][4];
      Evsrt_pp_numu   = MM_pp[epp*Nb_en_sp + k][5];
      Evsrt_pp_anumu  = MM_pp[epp*Nb_en_sp + k][6];
      Evsrt_pp_nutau  = MM_pp[epp*Nb_en_sp + k][7];
      Evsrt_pp_anutau = MM_pp[epp*Nb_en_sp + k][8];
      	    
      // Matrices definimus 
      
      MM_pp_ph(k, epp) = Evsrt_pp_ph;
      MM_pp_em(k, epp) = Evsrt_pp_em;
      MM_pp_ep(k, epp) = Evsrt_pp_ep;
      
      MM_pp_nue(k, epp) = Evsrt_pp_nue;
      MM_pp_anue(k, epp) = Evsrt_pp_anue;
      
      MM_pp_numu(k, epp) = Evsrt_pp_numu;
      MM_pp_anumu(k, epp) = Evsrt_pp_anumu;
      
      MM_pp_nutau(k, epp) = Evsrt_pp_nutau;
      MM_pp_anutau(k, epp) = Evsrt_pp_anutau;

      file_pp << range_E[k] << '\t' << Enu_sp_i << '\t' << Evsrt_pp_numu << endl; 
      
    }
	
  }

  MM_pp_ph.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_ph_E.bin");
  MM_pp_em.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_em_E.bin");
  MM_pp_ep.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_ep_E.bin");

  MM_pp_nue.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_nue_E.bin");
  MM_pp_anue.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_anue_E.bin");   
  MM_pp_numu.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_numu_E.bin");
  MM_pp_anumu.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_anumu_E.bin"); 
  MM_pp_nutau.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_nutau_E.bin");
  MM_pp_anutau.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_anutau_.bin");

  MM_pp_numu.save("./Data/"+pp_s_f+"/MM_"+pp_s_f+"_numu_E.dat", raw_ascii);

  auto end_MM = std::chrono::system_clock::now();
      
  std::chrono::duration<double> elapsed_seconds_MM = end_MM - start_MM;
  std::time_t end_time_MM = std::chrono::system_clock::to_time_t(end_MM);

  auto MM_time = localtime(&end_time_MM);
  auto MMtm  = std::put_time(MM_time, "%H:%M:%S ante diem %d-%m-%Y");
	  
  cout << endl << endl;
  cout << "Mātrices Mīgrātiōnis ē Pythia factae sunt." << endl << endl;
  cout << "Tempus intercessum: " << elapsed_seconds_MM.count() << " s = "
       << elapsed_seconds_MM.count()/60. << " min = "
       << elapsed_seconds_MM.count()/3600. << " h." << endl << endl
       << "Programma ab hōrā " << MMtm << " perfecit." << endl;
  cout << endl;
  
  return 0;
  
}
