TARGET = opencv_app
OBJECTS = $(wildcard *.cpp)

PKGS = gio-2.0 gio-unix-2.0 vdostream

CXXFLAGS += -Os -pipe -std=c++17
CXXFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags-only-I $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))

CXXFLAGS += -I$(SDKTARGETSYSROOT)/usr/include/opencv4
LDFLAGS = -L./lib -Wl,--no-as-needed,-rpath,'$$ORIGIN/lib'
LDLIBS += -lm -lopencv_aruco -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -lpthread -lSnowDepth

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ -o $@ ; \
	$(STRIP) --strip-unneeded $@

clean:
	$(RM) *.o $(TARGET) *.eap* *_LICENSE.txt pa*
