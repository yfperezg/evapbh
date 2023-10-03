//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//                                                                                                                                   //
//                                         Particulae Secondariae e foraminibus atris emissae                                        //
//                                                      Pythia 8 Computatio Facta                                                    //
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
#include <gsl/gsl_histogram.h>   // 1D Histograma
#include <gsl/gsl_histogram2d.h> // 2D Histograma
//
#include <armadillo>             // Bibliothecae Armadillo
//
#include "Pythia8/Pythia.h"      // Bibliothecae Pythiae
//
//
#define Nb_en_pp  250  // Numerus lacuum in energia particulae primariae
#define Nb_en_sp  500  // Numerus lacuum in energia particulae secondariae
//
//
using namespace Pythia8;
using namespace arma;

std::vector<double> rot_vec(double px, double py, double pz, double theta, double phi) {

  std::vector<double> s = {-pz*sin(theta) + cos(theta)*(px*cos(phi) + py*sin(phi)),
			   -px*sin(phi) + py*cos(phi),
			    pz*cos(theta) + sin(theta)*(px*cos(phi) + py*sin(phi))};
  return s;

}

//-----------------------------------------------------------------------------------//
//                           Programma Praecĭpŭum
//-----------------------------------------------------------------------------------//

 int main(int argc, char* argv[]) {

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

  ofstream file_pp, file_MM; string pp_s_f;

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

  file_pp.open("./Data/"+pp_s_f+"/primpar_"+pp_s_f+"_E.dat", ios::out);
  
  file_pp << scientific;  file_pp.precision(10);

  cout << scientific << std::showpoint;

  file_MM.open("./Data/"+pp_s_f+"/MM-"+pp_s_f+"_E.txt", ios::out);

  file_MM << scientific;  file_MM.precision(10);

  auto begin = std::chrono::system_clock::now();
  std::time_t begin_time = std::chrono::system_clock::to_time_t(begin);

  //cout << int(begin_time/100) << endl;
  
  const int NMM = Nb_en_sp*(Nb_en_pp+1);
  std::vector<std::vector<double>> MM_pp(NMM, std::vector<double>(9));
  
  // Particularum histogramae in energia anguloque
  
  size_t nx = Nb_en_sp;

  double range_E[Nb_en_sp + 1];
  
  double en_pp_min = log10(5.); // Log10@Etru min --- energia minima particulae primariae
  double en_pp_max = 5.; // Log10@Etru max --- energia maxima particulae primariae

  double en_sp_min = -6.; // Log10@Etru min --- energia minima particulae secondariae
  double en_sp_max = 5.; // Log10@Etru max --- --- energia minima particulae secondariae

  for(int i = 0; i <= Nb_en_sp; i++) range_E[i]  = pow(10., en_sp_min + (en_sp_max - en_sp_min)*i/(Nb_en_sp-1));
  

  // Histogramae particulae latae a colisionis

  // Photones
  gsl_histogram * h_eth_pp_ph = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_ph, range_E, nx+1);

  // Electrones
  gsl_histogram * h_eth_pp_em = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_em, range_E, nx+1);

  // Positrones
  gsl_histogram * h_eth_pp_ep = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_ep, range_E, nx+1);

  // Neutrinum electronicum
  gsl_histogram * h_eth_pp_nue = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_nue, range_E, nx+1);

  // Antineutrinum electronicum
  gsl_histogram * h_eth_pp_anue = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_anue, range_E, nx+1);

  // Neutrinum muonicum
  gsl_histogram * h_eth_pp_numu = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_numu, range_E, nx+1);

  // Antineutrinum muonicum
  gsl_histogram * h_eth_pp_anumu = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_anumu, range_E, nx+1);

  // Neutrinum tauonicum
  gsl_histogram * h_eth_pp_nutau = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_nutau, range_E, nx+1);

  // Antineutrinum muonicum
  gsl_histogram * h_eth_pp_anutau = gsl_histogram_alloc (nx);
  gsl_histogram_set_ranges(h_eth_pp_anutau, range_E, nx+1);
  

  // Generator Eventorum Pythiae factus
  Pythia pythia;
  pythia.readFile(argv[1]);

  string rs = "Random:seed = " + to_string(int(begin_time/100));
  //pythia.readString(rs);

  string decay;

  if (prim_par <= b){//Quarci

    pythia.readString("WeakSingleBoson:ffbar2ffbar(s:gmZ) = on");
    pythia.readString("23:onMode = off");

    decay = "23:onIfAny = " + prp_s;
  
    pythia.readString(decay);

  }
  
  else if (prim_par == t){// Top

    pythia.readString("Top:ffbar2ttbar(s:gmZ) = on");

  }
  
  else if (prim_par > t and prim_par <= Z){//leptones, gluones, photones, Zque
    
    pythia.readString("HiggsSM:ffbar2H = on");
    pythia.readString("25:onMode = off");

    decay = "25:onIfAny = " + prp_s;
  
    pythia.readString(decay);
    
  }

  else if (prim_par == W){//W
    
    pythia.readString("WeakDoubleBoson:ffbar2WW = on");
    
  }

  else pythia.readString("HiggsSM:ffbar2H = on"); // Higgs

  int nEvent = pythia.mode("Main:numberOfEvents");
  
  int stat;

  if (prim_par <= gam and prim_par != t) stat = 23;
  else stat = 22;

  double enCM; string enCM_s;
 
  for(int epp = 0; epp < Nb_en_pp; epp++){

  //int epp = 0;

    enCM = 2.*pow(10., en_pp_min + (en_pp_max - en_pp_min)*epp/(Nb_en_pp - 1)); // Massae Centrum Energia -> bis Energia particulae primariae

    cout << enCM/2. << endl;

    enCM_s = "Beams:eCM = " + to_string(enCM);
    
    pythia.readString(enCM_s);
    pythia.init();    
    
    // Indices eventorum e Pythia 
    
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
      
      if (!pythia.next()) continue;
      
      for (int i = 0; i < pythia.event.size(); i++){
	
	// ***************************************************************
	//    Elegimus particulas quae filae particulae primariae sunt
	// ***************************************************************
	
	// Photones
	if (pythia.event[i].isFinal() && pythia.event[i].id() == gam)    gsl_histogram_increment(h_eth_pp_ph, pythia.event[i].e());
	
	//Electrones
	if (pythia.event[i].isFinal() && pythia.event[i].id() == e)      gsl_histogram_increment(h_eth_pp_em, pythia.event[i].e());
	
	//Positrones
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -e)     gsl_histogram_increment(h_eth_pp_ep, pythia.event[i].e());
	
	//Neutrinum electronicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == nue)    gsl_histogram_increment(h_eth_pp_nue, pythia.event[i].e());
	
	// Antineutrinum electronicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -nue)   gsl_histogram_increment(h_eth_pp_anue, pythia.event[i].e());
	
	//Neutrinum muonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == numu)   gsl_histogram_increment(h_eth_pp_numu, pythia.event[i].e());
	
	//Antieutrinum muonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -numu)  gsl_histogram_increment(h_eth_pp_anumu, pythia.event[i].e());
	
	//Neutrinum tauonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == nutau)  gsl_histogram_increment(h_eth_pp_nutau, pythia.event[i].e());
	
	//Antineutrinum muonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -nutau) gsl_histogram_increment(h_eth_pp_anutau, pythia.event[i].e());
	
      }
      
    }

    pythia.stat();

    for(int k = 0; k < Nb_en_sp; k++)    
      {
	
	MM_pp[epp*Nb_en_sp + k][0] = gsl_histogram_get(h_eth_pp_ph,     k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][1] = gsl_histogram_get(h_eth_pp_em,     k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][2] = gsl_histogram_get(h_eth_pp_ep,     k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][3] = gsl_histogram_get(h_eth_pp_nue,    k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][4] = gsl_histogram_get(h_eth_pp_anue,   k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][5] = gsl_histogram_get(h_eth_pp_numu,   k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][6] = gsl_histogram_get(h_eth_pp_anumu,  k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][7] = gsl_histogram_get(h_eth_pp_nutau,  k)/nEvent/2.;
	MM_pp[epp*Nb_en_sp + k][8] = gsl_histogram_get(h_eth_pp_anutau, k)/nEvent/2.;
	
      }

    gsl_histogram_reset(h_eth_pp_ph);
    gsl_histogram_reset(h_eth_pp_em);    gsl_histogram_reset(h_eth_pp_ep);
    gsl_histogram_reset(h_eth_pp_nue);   gsl_histogram_reset(h_eth_pp_anue);
    gsl_histogram_reset(h_eth_pp_numu);  gsl_histogram_reset(h_eth_pp_anumu);
    gsl_histogram_reset(h_eth_pp_nutau); gsl_histogram_reset(h_eth_pp_anutau);

  }


  gsl_histogram_free(h_eth_pp_ph);
  gsl_histogram_free(h_eth_pp_em);    gsl_histogram_free(h_eth_pp_ep);
  gsl_histogram_free(h_eth_pp_nue);   gsl_histogram_free(h_eth_pp_anue);
  gsl_histogram_free(h_eth_pp_numu);  gsl_histogram_free(h_eth_pp_anumu);
  gsl_histogram_free(h_eth_pp_nutau); gsl_histogram_free(h_eth_pp_anutau);

  
  for(int m = 0; m < NMM; m++) {
    for(int n = 0; n < 9; n++){
      
	  file_MM << MM_pp[m][n] <<'\t' ;
    }

    file_MM << endl;

  }


  auto end = std::chrono::system_clock::now();
      
  std::chrono::duration<double> elapsed_seconds = end-begin;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  auto ed_time = localtime(&end_time);
  auto endtm  = std::put_time(ed_time, "%H:%M:%S ante diem %d-%m-%Y");

  cout << " " << endl;
  cout << "Matricēs e Pythia facta sunt. Tempus intercessum: " << elapsed_seconds.count() << " s = "
       << elapsed_seconds.count()/60. << " min = " << elapsed_seconds.count()/3600. << " h. Ab hōrā "
       << endtm << " perfecit" << endl << endl;

  cout << '\t' << endl;
  
  return 0;
  
}
