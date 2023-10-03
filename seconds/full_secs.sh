#bin/bash
clear
make secpbh; make rsecpbh << EOF
12
EOF
make secpbh; make rsecpbh << EOF
-12
EOF
make secpbh; make rsecpbh << EOF
14
EOF
make secpbh; make rsecpbh << EOF
-14
EOF
make secpbh; make rsecpbh << EOF
16
EOF
make secpbh; make rsecpbh << EOF
-16
EOF
make secpbh; make rsecpbh << EOF
22
EOF
