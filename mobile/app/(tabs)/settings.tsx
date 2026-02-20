import { useState, useEffect } from 'react';
import { StyleSheet, TextInput } from 'react-native';
import { Text, View } from '@/components/Themed';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { DEFAULT_BACKEND_BASE_URL } from '@/constants/config';

const BACKEND_URL_KEY = 'basketball_optimizer_backend_url';

export default function SettingsScreen() {
  const [url, setUrl] = useState(DEFAULT_BACKEND_BASE_URL);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(BACKEND_URL_KEY).then((stored) => {
      if (stored) setUrl(stored);
    });
  }, []);

  const saveUrl = () => {
    AsyncStorage.setItem(BACKEND_URL_KEY, url.trim()).then(() => {
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    });
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings</Text>
      <Text style={styles.label}>Backend base URL</Text>
      <TextInput
        style={styles.input}
        value={url}
        onChangeText={setUrl}
        placeholder={DEFAULT_BACKEND_BASE_URL}
        placeholderTextColor="#999"
        autoCapitalize="none"
        autoCorrect={false}
        onBlur={saveUrl}
      />
      <Text style={styles.hint}>
        Use your computer's LAN IP when testing on a device (e.g. http://192.168.1.100:8000).
        Tap outside the field to save.
      </Text>
      {saved && <Text style={styles.saved}>Saved.</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    paddingTop: 40,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 24,
  },
  label: {
    fontSize: 14,
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 12,
  },
  hint: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 16,
  },
  saved: {
    fontSize: 14,
    color: '#2f95dc',
  },
});
