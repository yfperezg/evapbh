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
#include <gsl/gsl_histogram2d.h> // 2D Histograma
//
#include <armadillo>             // Bibliothecae Armadillo
//
#include "Pythia8/Pythia.h"      // Bibliothecae Pythiae
//
//
#define Nb_en_pp  50 // Numerus lacuum in energia particulae primariae
#define Nb_en_sp  50 // Numerus lacuum in energia particulae secondariae
#define Nb_del    100 // Numerus lacuum in angulo relativo
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

  file_pp.open("./Data/"+pp_s_f+"/primpar_"+pp_s_f+".dat", ios::out);
  
  file_pp << scientific;  file_pp.precision(10);

  cout << scientific << std::showpoint;

  file_MM.open("./Data/"+pp_s_f+"/MM-"+pp_s_f+".txt", ios::out);

  file_MM << scientific;  file_MM.precision(10);

  auto begin = std::chrono::system_clock::now();
  std::time_t begin_time = std::chrono::system_clock::to_time_t(begin);

  //cout << int(begin_time/100) << endl;
  
  const int NMM = Nb_en_sp*(Nb_en_pp+1)*Nb_del;
  std::vector<std::vector<double>> MM_pp(NMM, std::vector<double>(9));
  
  // Particularum histogramae in energia anguloque
  
  size_t nx = Nb_en_sp;
  size_t ny = Nb_del;

  double range_E[Nb_en_sp + 1];
  double range_del[Nb_del + 1];

  
  double en_pp_min = 2.; // Log10@Etru min --- energia minima particulae primariae
  double en_pp_max = 6.; //log10(2.5e5);//6.; // Log10@Etru max --- energia maxima particulae primariae

  double en_sp_min = 0.;  // Log10@Etru min --- energia minima particulae secondariae
  double en_sp_max = 6.;  // Log10@Etru min --- energia minima particulae secondariae
  //double Den_sp    = 0.2; // Gradus logaritmicus -- energia particulae secondariae

  double delmin = -1.,  delmax = 1.; // Cos anguli relativi

  for(int i = 0; i <= Nb_en_sp; i++) range_E[i]  = pow(10., en_sp_min + (en_sp_max - en_sp_min)*i/Nb_en_sp);
  for(int i = 0; i <= Nb_del; i++)  range_del[i] = delmin + (delmax - delmin)*i/Nb_del; 

  // Histogramae particulae latae a colisionis

  // Photones
  gsl_histogram2d * h2d_eth_pp_ph = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_ph, range_E, nx+1, range_del, ny+1);

  // Electrones
  gsl_histogram2d * h2d_eth_pp_em = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_em, range_E, nx+1, range_del, ny+1);

  // Positrones
  gsl_histogram2d * h2d_eth_pp_ep = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_ep, range_E, nx+1, range_del, ny+1);

  // Neutrinum electronicum
  gsl_histogram2d * h2d_eth_pp_nue = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_nue, range_E, nx+1, range_del, ny+1);

  // Antineutrinum electronicum
  gsl_histogram2d * h2d_eth_pp_anue = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_anue, range_E, nx+1, range_del, ny+1);

  // Neutrinum muonicum
  gsl_histogram2d * h2d_eth_pp_numu = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_numu, range_E, nx+1, range_del, ny+1);

  // Antineutrinum muonicum
  gsl_histogram2d * h2d_eth_pp_anumu = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_anumu, range_E, nx+1, range_del, ny+1);

  // Neutrinum tauonicum
  gsl_histogram2d * h2d_eth_pp_nutau = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_nutau, range_E, nx+1, range_del, ny+1);

  // Antineutrinum muonicum
  gsl_histogram2d * h2d_eth_pp_anutau = gsl_histogram2d_alloc (nx, ny);
  gsl_histogram2d_set_ranges(h2d_eth_pp_anutau, range_E, nx+1, range_del, ny+1);
  

  // Generator Eventorum Pythiae factus
  Pythia pythia;
  pythia.readFile(argv[1]);

  string rs = "Random:seed = " + to_string(int(begin_time/100));
  pythia.readString(rs);

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
 
  for(int epp = 0; epp <= Nb_en_pp; epp++){

  //int epp = 0;

    enCM = 2.*pow(10., en_pp_min + (en_pp_max - en_pp_min)*epp/Nb_en_pp); // Massae Centrum Energia -> bis Energia particulae primariae

    cout << enCM/2. << endl;

    enCM_s = "Beams:eCM = " + to_string(enCM);
    
    pythia.readString(enCM_s);
    pythia.init();    
    
    // Indices eventorum e Pythia 
    
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
      
      if (!pythia.next()) continue;

      double theta_ppar = 0., phi_ppar = 0.;

      std::vector<int> id_dtg_pp, id_dtg_app;
      
      for (int i = 0; i < pythia.event.size(); i++){
	
	// ***************************************************************
	//    Elegimus particulas quae filae particulae primariae sunt
	// ***************************************************************

	
	if (pythia.event[i].id() == prim_par and pythia.event[i].statusAbs() == stat and pythia.event[i].pz() != 0.){

	  // Anguli definimus ut quantitatem motum particulae primariae in axe z sit
	  
	  theta_ppar = acos(pythia.event[i].pz()/sqrt(pythia.event[i].px()*pythia.event[i].px() +
						      pythia.event[i].py()*pythia.event[i].py() +
						      pythia.event[i].pz()*pythia.event[i].pz()));
	  phi_ppar   = sign(pythia.event[i].py())*acos(pythia.event[i].px()/sqrt(pythia.event[i].px()*pythia.event[i].px() +
										 pythia.event[i].py()*pythia.event[i].py()));
	  }
	
	// Photones
	if (pythia.event[i].isFinal() && pythia.event[i].id() == gam){
	  
	  std::vector<double> rot_ph = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
					       theta_ppar, phi_ppar);
	  
	  double cos_del_ph = rot_ph[2]/sqrt(rot_ph[0]*rot_ph[0] + rot_ph[1]*rot_ph[1] + rot_ph[2]*rot_ph[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_ph, pythia.event[i].e(), abs(cos_del_ph));
	  
	}
	
	//Electrones
	if (pythia.event[i].isFinal() && pythia.event[i].id() == e){
	  
	  std::vector<double> rot_em = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
					       theta_ppar, phi_ppar);
	  
	  double cos_del_em = rot_em[2]/sqrt(rot_em[0]*rot_em[0] + rot_em[1]*rot_em[1] + rot_em[2]*rot_em[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_em, pythia.event[i].e(), abs(cos_del_em));
	  
	}
	
	//Positrones
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -e){
	  
	  std::vector<double> rot_ep = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
					       theta_ppar, phi_ppar);
	  
	  double cos_del_ep = rot_ep[2]/sqrt(rot_ep[0]*rot_ep[0] + rot_ep[1]*rot_ep[1] + rot_ep[2]*rot_ep[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_ep, pythia.event[i].e(), abs(cos_del_ep));
	  
	}
	
	//Neutrinum electronicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == nue){
	  
	  std::vector<double> rot_nue = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
						theta_ppar, phi_ppar);
	  
	  double cos_del_nue = rot_nue[2]/sqrt(rot_nue[0]*rot_nue[0] + rot_nue[1]*rot_nue[1] + rot_nue[2]*rot_nue[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_nue, pythia.event[i].e(), abs(cos_del_nue));
	  
	}
	
	// Antineutrinum electronicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -nue){
	  
	  std::vector<double> rot_anue = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
						 theta_ppar, phi_ppar);
	  
	  double cos_del_anue = rot_anue[2]/sqrt(rot_anue[0]*rot_anue[0] + rot_anue[1]*rot_anue[1] + rot_anue[2]*rot_anue[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_anue, pythia.event[i].e(), abs(cos_del_anue));
	  
	}
	
	//Neutrinum muonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == numu){
	  
	  std::vector<double> rot_numu = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
						 theta_ppar, phi_ppar);
	  
	  double cos_del_numu = rot_numu[2]/sqrt(rot_numu[0]*rot_numu[0] + rot_numu[1]*rot_numu[1] + rot_numu[2]*rot_numu[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_numu, pythia.event[i].e(), abs(cos_del_numu));
	  
	}
	
	//Antieutrinum muonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -numu){
	  
	  std::vector<double> rot_anumu = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
						  theta_ppar, phi_ppar);
	  
	  double cos_del_anumu = rot_anumu[2]/sqrt(rot_anumu[0]*rot_anumu[0] + rot_anumu[1]*rot_anumu[1] + rot_anumu[2]*rot_anumu[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_anumu, pythia.event[i].e(), abs(cos_del_anumu));
	  
	}
	
	//Neutrinum tauonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == nutau){
	  
	  std::vector<double> rot_nutau = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
						  theta_ppar, phi_ppar);
	  
	  double cos_del_nutau = rot_nutau[2]/sqrt(rot_nutau[0]*rot_nutau[0] + rot_nutau[1]*rot_nutau[1] + rot_nutau[2]*rot_nutau[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_nutau, pythia.event[i].e(), abs(cos_del_nutau));
	  
	}
	
	//Antineutrinum muonicum
	if (pythia.event[i].isFinal() && pythia.event[i].id() == -nutau){
	  
	  std::vector<double> rot_anutau = rot_vec(pythia.event[i].px(), pythia.event[i].py(), pythia.event[i].pz(),
						   theta_ppar, phi_ppar);
	  
	  double cos_del_anutau = rot_anutau[2]/sqrt(rot_anutau[0]*rot_anutau[0] + rot_anutau[1]*rot_anutau[1] + rot_anutau[2]*rot_anutau[2]);
	  
	  gsl_histogram2d_increment(h2d_eth_pp_anutau, pythia.event[i].e(), abs(cos_del_anutau));
	  
	}
	
      }
      
    }

    pythia.stat();

    for(int k = 0; k < Nb_en_sp; k++)    
      for(int l = 0; l < Nb_del; l++)
	{
	  
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][0] = gsl_histogram2d_get(h2d_eth_pp_ph, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][1] = gsl_histogram2d_get(h2d_eth_pp_em, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][2] = gsl_histogram2d_get(h2d_eth_pp_ep, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][3] = gsl_histogram2d_get(h2d_eth_pp_nue, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][4] = gsl_histogram2d_get(h2d_eth_pp_anue, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][5] = gsl_histogram2d_get(h2d_eth_pp_numu, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][6] = gsl_histogram2d_get(h2d_eth_pp_anumu, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][7] = gsl_histogram2d_get(h2d_eth_pp_nutau, k, l)/nEvent/2.;
	  MM_pp[epp*Nb_en_sp*Nb_del + k*Nb_del + l][8] = gsl_histogram2d_get(h2d_eth_pp_anutau, k, l)/nEvent/2.;
	   
      }

    gsl_histogram2d_reset(h2d_eth_pp_ph);
    gsl_histogram2d_reset(h2d_eth_pp_em);    gsl_histogram2d_reset(h2d_eth_pp_ep);
    gsl_histogram2d_reset(h2d_eth_pp_nue);   gsl_histogram2d_reset(h2d_eth_pp_anue);
    gsl_histogram2d_reset(h2d_eth_pp_numu);  gsl_histogram2d_reset(h2d_eth_pp_anumu);
    gsl_histogram2d_reset(h2d_eth_pp_nutau); gsl_histogram2d_reset(h2d_eth_pp_anutau);

  }


  gsl_histogram2d_free(h2d_eth_pp_ph);
  gsl_histogram2d_free(h2d_eth_pp_em);    gsl_histogram2d_free(h2d_eth_pp_ep);
  gsl_histogram2d_free(h2d_eth_pp_nue);   gsl_histogram2d_free(h2d_eth_pp_anue);
  gsl_histogram2d_free(h2d_eth_pp_numu);  gsl_histogram2d_free(h2d_eth_pp_anumu);
  gsl_histogram2d_free(h2d_eth_pp_nutau); gsl_histogram2d_free(h2d_eth_pp_anutau);

  
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
