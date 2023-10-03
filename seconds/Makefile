#---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#      Epistola          : Makefile                                                                 #
#                                                                                                   #
#      Dēscriptiō        : Particulae secondariae e foraminibus alteris emissae Pythia computatae   #
#                                                                                                   #
#      Diēs fōrmātiōnis  : pridie Idus Apriles XMMXXIII E.H. -- MMDCCLXXVI A.U.C.                   #
#                                                                                                   #
#      Ultima recognitio : Ante diem XII Kalendas Maias XMMXXIII E.H. -- MMDCCLXXVI A.U.C.          #
#                                                                                                   #  
#      Auctor            : Yuber F Perez-Gonzalez                                                   #
#                                                                                                   #  
#---------------------------------------------------------------------------------------------------#

# SHELL recta confirmāmus 

SHELL = /bin/sh
MAKE  = gmake

# Directoriī radicum dēfīnīmus 

NuE_ROOT := .
NuE_DAT  := ./data

# Arsterm dēfīnīmus       

BIN_DIR := $(NuE_ROOT)

# Directoriī principalēs dēfīnīmus

SRCDIR := $(NuE_ROOT)

# Directoriī auxiliārēs dēfīnīmus

AUXDIR := $(NuE_ROOT)/../Auxfiles

# Directoriī Pythiae dēfīnīmus

PYTDIR := /home/yparez-gonzalez/Documents/PROGRAMMA/pythia8309
PYT_LIB = $(PYTDIR)/lib
PYT_INC = $(PYTDIR)/include
PYTHIA  = $(PYT_LIB)/libpythia8$(LIB_SUFFIX)

#++++++++++++++++++++++++++++++++++++++++++++#
#    
#++++++++++++++++++++++++++++++++++++++++++++#

src_files  = $(wildcard $(SRCDIR)/pythia_secs.cc)
cmnd_files = $(wildcard $(SRCDIR)/pythia_secs.cmnd)
log_files  = $(wildcard $(SRCDIR)/pythia_secs.log)

main_obj_files += $(src_files:%.cpp=%.o)
main_ofiles    += $(notdir $(main_obj_files))

# Nomen tabulāriōrum executōrum

execfit_pyth:= $(BIN_DIR)/pythia_secs

#++++++++++++++++++++++++++++++++++++++++++++#
#    
#++++++++++++++++++++++++++++++++++++++++++++#

src_E_files  = $(wildcard $(SRCDIR)/pythia_secs_E.cc)
cmnd_E_files = $(wildcard $(SRCDIR)/pythia_secs_E.cmnd)
log_E_files  = $(wildcard $(SRCDIR)/pythia_secs_E.log)

main_obj_files_E += $(src_E_files:%.cpp=%.o)
main_ofiles_E    += $(notdir $(main_obj_files_E))

# Nomen tabulāriōrum executōrum

execfit_E_pyth:= $(BIN_DIR)/pythia_secs_E

#++++++++++++++++++++++++++++++++++++++++++++#
#    
#++++++++++++++++++++++++++++++++++++++++++++#

MM_src_files  = $(wildcard $(SRCDIR)/MM.cpp)
MM_main_obj_files += $(MM_src_files:%.cpp=%.o)
MM_main_ofiles    += $(notdir $(MM_main_obj_files))

# Nomen tabulāriōrum executōrum

execfit_MM:= $(BIN_DIR)/MM

#++++++++++++++++++++++++++++++++++++++++++++#
#    
#++++++++++++++++++++++++++++++++++++++++++++#

MM_src_E_files  = $(wildcard $(SRCDIR)/MM_E.cpp)
MM_main_obj_files_E += $(MM_src_E_files:%.cpp=%.o)
MM_main_ofiles_E    += $(notdir $(MM_main_obj_files_E))

# Nomen tabulāriōrum executōrum

execfit_MM_E:= $(BIN_DIR)/MM_E

#++++++++++++++++++++++++++++++++++++++++++++#
#    
#++++++++++++++++++++++++++++++++++++++++++++#

sec_src_files  = $(wildcard $(SRCDIR)/secpbh.cpp)
sec_main_obj_files += $(sec_src_files:%.cpp=%.o)
sec_main_ofiles    += $(notdir $(sec_main_obj_files))

# Nomen tabulāriōrum executōrum

execfit_spbh:= $(BIN_DIR)/secpbh

#++++++++++++++++++++++++++++++++++++++++++++#
#    
#++++++++++++++++++++++++++++++++++++++++++++#

sec_E_src_files  = $(wildcard $(SRCDIR)/secpbh_E.cpp)
sec_E_main_obj_files += $(sec_E_src_files:%.cpp=%.o)
sec_E_main_ofiles    += $(notdir $(sec_E_main_obj_files))

# Nomen tabulāriōrum executōrum

execfit_spbh_E:= $(BIN_DIR)/secpbh_E


#-------------------------------------------------------#
#-------          Vexilla pro compositio         -------#
#-------------------------------------------------------#

# Vexilla dēfīnīmus -- Fortran

FC      = gfortran
FFLAGS  += -o

# Vexilla dēfīnīmus -- C++

CC      = g++

CXX_COMMON := -std=c++11 -I/usr/local/include -I$(PYT_INC) $(CXX_COMMON) $(GZIP_LIB) -O3 -W -Wall -Wshadow -fPIC
CXX_COMMON += -L/usr/local/lib -larmadillo -L$(PYT_LIB) -Wl,-rpath,$(PYT_LIB) -lpythia8 -ldl -lgsl -lgslcblas -lm -pedantic-errors 

.PHONY : all pytmm epytm scpbh dspbh  mm hemm clean cleanall help
.PHONY : sclean

all: $(execfit_pyth) $(execfit_spbh)
pytmm: $(execfit_pyth)
epytm: $(execfit_E_pyth)
scpbh: $(execfit_spbh)
mm:    $(execfit_MM)
hemm:  $(execfit_MM_E)
dspbh: $(execfit_spbh_E)

$(execfit_pyth): $(main_obj_files)
	@echo '=========================================================='
	@echo ' Tabulārium fūnctiōnis compōnimus ... '
	@echo '    '
	@rm -f  $@
	@$(CC) $@.cc -o $@ $(CXX_COMMON)
	@echo '    '
	@echo ' Factum est. '
	@echo '=========================================================='

$(execfit_E_pyth): $(main_obj_files_E)
	@echo '=========================================================='
	@echo ' Tabulārium fūnctiōnis compōnimus ... '
	@echo '    '
	@rm -f  $@
	@$(CC) $@.cc -o $@ $(CXX_COMMON)
	@echo '    '
	@echo ' Factum est. '
	@echo '=========================================================='

$(execfit_spbh): $(sec_main_obj_files)
	@echo '=========================================================='
	@echo ' Tabulārium fūnctiōnis compōnimus ... '
	@echo '    '
	@rm -f  $@
	@$(CC) $@.cpp -o $@ $(CXX_COMMON)
	@echo '    '
	@echo ' Factum est. '
	@echo '=========================================================='

$(execfit_spbh_E): $(sec_E_main_obj_files)
	@echo '=========================================================='
	@echo ' Tabulārium fūnctiōnis compōnimus ... '
	@echo '    '
	@rm -f  $@
	@$(CC) $@.cpp -o $@ $(CXX_COMMON)
	@echo '    '
	@echo ' Factum est. '
	@echo '=========================================================='


$(execfit_MM): $(MM_main_obj_files)
	@echo '=========================================================='
	@echo ' Tabulārium fūnctiōnis compōnimus ... '
	@echo '    '
	@rm -f  $@
	@$(CC) $@.cpp -o $@ $(CXX_COMMON)
	@echo '    '
	@echo ' Factum est. '
	@echo '=========================================================='

$(execfit_E_MM): $(MM_main_obj_files_E)
	@echo '=========================================================='
	@echo ' Tabulārium fūnctiōnis compōnimus ... '
	@echo '    '
	@rm -f  $@
	@$(CC) $@.cpp -o $@ $(CXX_COMMON)
	@echo '    '
	@echo ' Factum est. '
	@echo '=========================================================='

clean :
	@find . -name "*.bak" -exec rm -f '{}' ';'
	@find . -name "*~" -exec rm -f '{}' ';'
	@find . -name "#*#"             -exec rm -f '{}' ';'
	@echo
	@echo Cleaning up *.o *.d *~ core
	@echo
	@-rm -f *.o  *~ core *.d
	@echo ' Factum est.'

cleanall : 
	@echo '**********************************************************'
	@echo ' Omnia purgāmus : '
	@find . -wholename "*.bak" -exec rm -f '{}' ';'
	@find . -wholename "*~" -exec rm -f '{}' ';'
	@find . -wholename "*.log"           -exec rm -f '{}' ';'
	@find . -wholename "../Auxfiles/*.o" -exec rm -f '{}' ';'
	@find . -wholename "*.out"           -exec rm -f '{}' ';'
	@find . -wholename "core"            -exec rm -f '{}' ';'
	@find . -wholename "#*#"             -exec rm -f '{}' ';'
	@-rm -f *.o ../Auxfiles/*.o
	@echo ' Factum est.'
	@echo '**********************************************************'

rpythmm :
	@echo '**********************************************************'	
	@echo ' Operātur ... '	
	@echo '    '
	@time -p $(execfit_pyth) $(cmnd_files)
#> $(log_files)
	@echo '	   '
	@echo ' Factum est. '
	@echo '**********************************************************'

repytmm :
	@echo '**********************************************************'	
	@echo ' Operātur ... '	
	@echo '    '
	@time -p $(execfit_E_pyth) $(cmnd_E_files)
#> $(log_files)
	@echo '	   '
	@echo ' Factum est. '
	@echo '**********************************************************'

rsecpbh :
	@echo '**********************************************************'	
	@echo ' Operātur ... '	
	@echo '    '
	@time -p $(execfit_spbh) 
	@echo '	   '
	@echo ' Factum est. '
	@echo '**********************************************************'

rmm :
	@echo '**********************************************************'	
	@echo ' Operātur ... '	
	@echo '    '
	@time -p $(execfit_MM) 
	@echo '	   '
	@echo ' Factum est. '
	@echo '**********************************************************'

rhemm :
	@echo '**********************************************************'	
	@echo ' Operātur ... '	
	@echo '    '
	@time -p $(execfit_MM_E) 
	@echo '	   '
	@echo ' Factum est. '
	@echo '**********************************************************'


rescpbh :
	@echo '**********************************************************'	
	@echo ' Operātur ... '	
	@echo '    '
	@time -p $(execfit_spbh_E) 
	@echo '	   '
	@echo ' Factum est. '
	@echo '**********************************************************'


help:
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	@echo ' '
	@echo '  Optionēs possibilēs sunt'
	@echo '  ========================  '
	@echo 
	@echo '  make secpbh   -- tabulāria fūnctiōnis particularum secondarum e foraminibus creatur'
	@echo '  make rsecpbh  -- tabulāria operatur'
	@echo '  make clean    -- tanta tabulāria purgāmus '
	@echo '  make cleanall -- omnia tabulāria purgāmus '
	@echo ' '
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
