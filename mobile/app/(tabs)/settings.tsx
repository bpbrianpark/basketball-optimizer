import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Animated,
  StatusBar,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { DEFAULT_BACKEND_BASE_URL } from '@/constants/config';
import { theme } from '@/constants/theme';

const BACKEND_URL_KEY = 'basketball_optimizer_backend_url';

export default function SettingsScreen() {
  const [url, setUrl] = useState(DEFAULT_BACKEND_BASE_URL);
  const [focused, setFocused] = useState(false);
  const savedOpacity = useRef(new Animated.Value(0)).current;
  const headerFade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(headerFade, { toValue: 1, duration: 500, useNativeDriver: true }).start();
    AsyncStorage.getItem(BACKEND_URL_KEY).then((stored) => {
      if (stored) setUrl(stored);
    });
  }, []);

  const saveUrl = () => {
    AsyncStorage.setItem(BACKEND_URL_KEY, url.trim()).then(() => {
      Animated.sequence([
        Animated.timing(savedOpacity, { toValue: 1, duration: 250, useNativeDriver: true }),
        Animated.timing(savedOpacity, { toValue: 0, duration: 250, delay: 1600, useNativeDriver: true }),
      ]).start();
    });
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={theme.bg} />

      {/* Header */}
      <Animated.View style={[styles.header, { opacity: headerFade }]}>
        <Text style={styles.eyebrow}>CONFIG</Text>
        <Text style={styles.title}>SETTINGS</Text>
        <View style={styles.accentBar} />
      </Animated.View>

      {/* Backend URL section */}
      <Text style={styles.sectionLabel}>BACKEND CONNECTION</Text>
      <View style={styles.card}>
        <Text style={styles.fieldLabel}>BASE URL</Text>
        <TextInput
          style={[styles.input, focused && styles.inputFocused]}
          value={url}
          onChangeText={setUrl}
          placeholder={DEFAULT_BACKEND_BASE_URL}
          placeholderTextColor={theme.textMuted}
          autoCapitalize="none"
          autoCorrect={false}
          onFocus={() => setFocused(true)}
          onBlur={() => {
            setFocused(false);
            saveUrl();
          }}
        />
        <Text style={styles.hint}>
          Use your machine's LAN IP when testing on a physical device.{'\n'}
          Example: http://192.168.1.100:8000
        </Text>
      </View>

      {/* Save button */}
      <TouchableOpacity style={styles.saveBtn} onPress={saveUrl} activeOpacity={0.82}>
        <Text style={styles.saveBtnText}>SAVE SETTINGS</Text>
      </TouchableOpacity>

      {/* Saved feedback */}
      <Animated.View style={[styles.savedWrap, { opacity: savedOpacity }]}>
        <Text style={styles.savedText}>✓  SAVED</Text>
      </Animated.View>

      {/* Status row */}
      <View style={styles.statusCard}>
        <Text style={styles.sectionLabel}>CURRENT TARGET</Text>
        <View style={styles.statusRow}>
          <View style={styles.statusDot} />
          <Text style={styles.statusUrl} numberOfLines={1}>
            {url || DEFAULT_BACKEND_BASE_URL}
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.bg,
    padding: 24,
    paddingTop: 48,
  },
  header: {
    marginBottom: 36,
  },
  eyebrow: {
    fontSize: 11,
    letterSpacing: 5,
    color: theme.accent,
    fontFamily: 'SpaceMono',
    marginBottom: 6,
  },
  title: {
    fontSize: 40,
    fontWeight: '900',
    color: theme.textPrimary,
    letterSpacing: 2,
  },
  accentBar: {
    width: 40,
    height: 3,
    backgroundColor: theme.accent,
    marginTop: 12,
    borderRadius: 2,
  },
  sectionLabel: {
    fontSize: 10,
    letterSpacing: 4,
    color: theme.textSecondary,
    marginBottom: 12,
  },
  card: {
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 4,
    padding: 20,
    marginBottom: 16,
  },
  fieldLabel: {
    fontSize: 10,
    letterSpacing: 3,
    color: theme.textSecondary,
    marginBottom: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 4,
    paddingVertical: 12,
    paddingHorizontal: 14,
    fontSize: 13,
    color: theme.textPrimary,
    backgroundColor: theme.bg,
    fontFamily: 'SpaceMono',
    marginBottom: 12,
  },
  inputFocused: {
    borderColor: theme.accent,
  },
  hint: {
    fontSize: 11,
    color: theme.textSecondary,
    lineHeight: 18,
    fontFamily: 'SpaceMono',
  },
  saveBtn: {
    backgroundColor: theme.accent,
    paddingVertical: 16,
    borderRadius: 4,
    alignItems: 'center',
    marginBottom: 8,
  },
  saveBtnText: {
    fontSize: 12,
    fontWeight: '800',
    color: theme.bg,
    letterSpacing: 3,
  },
  savedWrap: {
    alignSelf: 'center',
    paddingVertical: 10,
    marginBottom: 24,
  },
  savedText: {
    fontSize: 11,
    letterSpacing: 4,
    color: theme.success,
    fontFamily: 'SpaceMono',
  },
  statusCard: {
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 4,
    padding: 16,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.accent,
    flexShrink: 0,
  },
  statusUrl: {
    fontSize: 12,
    color: theme.textPrimary,
    fontFamily: 'SpaceMono',
    flex: 1,
  },
});
