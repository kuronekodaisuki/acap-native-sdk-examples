PROG1	= vdo_larod
OBJS1	= $(PROG1).cpp
PROGS	= $(PROG1)

PKGS = gio-2.0 vdostream gio-unix-2.0 liblarod cairo

CXXFLAGS += -Iinclude -I/axis/opencv/usr/include
CXXFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))
LDLIBS += -lm -laxoverlay -L /axis/opencv/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio

# This example is based on larod version 3
CXXFLAGS += -Wall -DLAROD_API_VERSION_3

all: $(PROGS)

$(PROG1): $(OBJS1)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f $(PROGS) *.o *.eap
